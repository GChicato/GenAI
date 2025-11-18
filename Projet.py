import os
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader

# 🔥 Langfuse remplace directement le client OpenAI
from langfuse.openai import OpenAI


# ============================================================
# 1. Config Streamlit
# ============================================================

def configure_page() -> None:
    st.set_page_config(
        page_title="Invoice Vision Dashboard",
        page_icon="📊",
        layout="wide",
    )


def render_header() -> None:
    st.title("📊 Invoice Vision Dashboard")
    st.markdown(
        """
Agent qui prend des **factures en PDF ou images**, en extrait les infos
et construit automatiquement un **dashboard de dépenses** + un **suivi comptable (factures payées / non payées)**.

Fonctionnalités :
- Lecture de factures PDF (texte) ou images (vision)
- Extraction automatique : fournisseur, date, montants, devise...
- Suivi comptable : facture payée / non, montant déjà payé, reste à payer
- Agrégation des montants par fournisseur, par mois
- Visualisation graphique (bar charts, camembert)

Backend : **API OpenAI (GPT-4.1-mini vision + GPT-4o-mini texte)**, tracée via **Langfuse**.
"""
    )


# ============================================================
# 2. OpenAI client
# ============================================================

def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def build_openai_client(api_key: str) -> OpenAI:
    # Langfuse récupère SECRET/PUBLIC/BASE_URL via variables d’environnement
    return OpenAI(api_key=api_key)


# ============================================================
# 3. Lecture fichiers (PDF / images)
# ============================================================

def detect_file_type(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        return "pdf"
    if uploaded_file.type.startswith("image/"):
        return "image"
    return "unknown"


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    temp_dir = Path("tmp_pdf")
    temp_dir.mkdir(exist_ok=True)
    pdf_path = temp_dir / "current.pdf"
    pdf_path.write_bytes(pdf_bytes)

    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text).strip()


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ============================================================
# 4. Agent vision : extraire texte d’une facture image
# ============================================================

def extract_text_from_image_with_gpt(client: OpenAI, image_b64: str) -> str:
    """
    Utilise gpt-4o-mini en mode vision pour lire la facture
    et renvoyer un texte brut exploitable.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        name="invoice_vision_extract",
        metadata={"component": "invoice_dashboard", "stage": "vision_extract"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui lit des factures sur des images et "
                    "transcris toutes les informations utiles (fournisseur, adresse, "
                    "date, lignes d'articles, montants, TVA) en texte brut."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Lis cette facture et renvoie UNIQUEMENT le texte brut "
                            "structuré, sans commentaire."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        temperature=0.0,
    )

    return completion.choices[0].message.content or ""


def get_invoice_text_for_file(client: OpenAI, uploaded_file) -> str:
    file_type = detect_file_type(uploaded_file)

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    if file_type == "pdf":
        return extract_text_from_pdf_bytes(file_bytes)

    if file_type == "image":
        b64 = encode_image_to_base64(file_bytes)
        return extract_text_from_image_with_gpt(client, b64)

    return ""


# ============================================================
# 5. Agent texte : structurer la facture en JSON
# ============================================================

def build_structuring_prompt(invoice_text: str) -> str:
    return f"""
Tu es un assistant expert en facturation.

On te donne le texte brut d'une **facture**.
Extrais un JSON avec exactement cette structure :

{{
  "vendor_name": "...",
  "invoice_number": "...",
  "invoice_date": "...",
  "currency": "...",
  "total_amount": 0.0,
  "tax_amount": 0.0,
  "line_items": [
    {{
      "description": "...",
      "quantity": 1,
      "unit_price": 0.0,
      "line_total": 0.0,
      "category": "..."
    }}
  ]
}}

Contraintes :
- Montants = nombres (float).
- "invoice_date" au format YYYY-MM-DD si possible, sinon laisse la date telle quelle.
- "currency" = code (EUR, USD, GBP...) si identifiable.
- "category" = un mot ou courte expression.

RENVOIE UNIQUEMENT ce JSON, sans texte autour, sans explication.

Texte de la facture :
\"\"\"{invoice_text}\"\"\"
"""


def structure_invoice_with_gpt(
    client: OpenAI,
    invoice_text: str,
    file_name: str,
) -> Dict[str, Any]:

    prompt = build_structuring_prompt(invoice_text)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        name="invoice_structuring",
        metadata={
            "component": "invoice_dashboard",
            "stage": "structured_json",
            "file_name": file_name,
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant expert en extraction de données de factures. "
                    "Tu réponds en JSON strict."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    content = completion.choices[0].message.content or ""

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    # Méta-infos
    data["file_name"] = file_name

    # Champs complémentaires
    data.setdefault("is_paid", False)
    data.setdefault("amount_paid", 0.0)

    return data


# ============================================================
# 6. Agrégation en DataFrame
# ============================================================

def to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def invoices_to_dataframe(invoices: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for inv in invoices:
        rows.append(
            {
                "file_name": inv.get("file_name", ""),
                "vendor_name": inv.get("vendor_name", "Inconnu"),
                "invoice_number": inv.get("invoice_number", ""),
                "invoice_date": inv.get("invoice_date", ""),
                "currency": inv.get("currency", ""),
                "total_amount": to_float(inv.get("total_amount", 0.0)),
                "tax_amount": to_float(inv.get("tax_amount", 0.0)),
                "is_paid": bool(inv.get("is_paid", False)),
                "amount_paid": to_float(inv.get("amount_paid", 0.0)),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df.loc[df["is_paid"] & (df["amount_paid"] <= 0.01), "amount_paid"] = df["total_amount"]

    df["amount_paid"] = df["amount_paid"].clip(lower=0)
    df["amount_paid"] = df[["amount_paid", "total_amount"]].min(axis=1)
    df["remaining_amount"] = (df["total_amount"] - df["amount_paid"]).clip(lower=0)

    if "invoice_date" in df.columns:
        df["invoice_date_parsed"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        df["invoice_month"] = df["invoice_date_parsed"].dt.to_period("M").astype(str)

    return df


# ============================================================
# 7. Dashboard + Suivi comptable
# ============================================================

def render_dashboard(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("Aucune facture à afficher.")
        return

    st.subheader("📌 Suivi de paiement des factures")

    edited_df = st.data_editor(
        df,
        key="invoice_editor",
        column_config={
            "file_name": st.column_config.TextColumn("Fichier", disabled=True),
            "vendor_name": st.column_config.TextColumn("Fournisseur", disabled=True),
            "invoice_number": st.column_config.TextColumn("N° facture", disabled=True),
            "invoice_date": st.column_config.TextColumn("Date facture", disabled=True),
            "currency": st.column_config.TextColumn("Devise", disabled=True),
            "total_amount": st.column_config.NumberColumn("Montant total", disabled=True),
            "tax_amount": st.column_config.NumberColumn("TVA", disabled=True),
            "is_paid": st.column_config.CheckboxColumn("Facture payée ?"),
            "amount_paid": st.column_config.NumberColumn("Montant déjà payé", disabled=True),
            "remaining_amount": st.column_config.NumberColumn("Reste à payer", disabled=True),
            "invoice_month": st.column_config.TextColumn("Mois", disabled=True),
        },
        hide_index=True,
    )

    edited_df["is_paid"] = edited_df["is_paid"].fillna(False).astype(bool)
    edited_df["amount_paid"] = 0.0
    edited_df.loc[edited_df["is_paid"], "amount_paid"] = edited_df["total_amount"]
    edited_df["remaining_amount"] = (edited_df["total_amount"] - edited_df["amount_paid"]).clip(lower=0)

    st.session_state["invoices_df"] = edited_df.copy()

    st.subheader("📈 Synthèse comptable")
    col1, col2, col3 = st.columns(3)

    total_due = edited_df["total_amount"].sum()
    total_paid = edited_df["amount_paid"].sum()
    total_remaining = edited_df["remaining_amount"].sum()

    col1.metric("Total facturé", f"{total_due:,.2f}")
    col2.metric("Déjà payé", f"{total_paid:,.2f}")
    col3.metric("Reste à payer", f"{total_remaining:,.2f}")

    st.markdown(
        f"""
- **Nombre de factures :** {len(edited_df)}  
- **Factures payées :** {int(edited_df['is_paid'].sum())}  
- **Factures non payées :** {len(edited_df) - int(edited_df['is_paid'].sum())}
"""
    )

    unpaid_df = edited_df[~edited_df["is_paid"]].copy()
    if not unpaid_df.empty:
        st.subheader("📌 Factures encore à payer")
        st.dataframe(
            unpaid_df[
                [
                    "file_name",
                    "vendor_name",
                    "invoice_number",
                    "invoice_date",
                    "currency",
                    "total_amount",
                    "remaining_amount",
                ]
            ],
            use_container_width=True,
        )

    st.subheader("🧾 Tableau des factures (avec suivi)")
    st.dataframe(
        edited_df[
            [
                "file_name",
                "vendor_name",
                "invoice_number",
                "invoice_date",
                "currency",
                "total_amount",
                "amount_paid",
                "remaining_amount",
                "is_paid",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("🏢 Dépenses par fournisseur (total facturé)")
    vendor_df = edited_df.groupby("vendor_name", as_index=False)["total_amount"].sum()
    st.plotly_chart(
        px.bar(vendor_df, x="vendor_name", y="total_amount"),
        use_container_width=True,
    )

    st.subheader("💸 Reste à payer par fournisseur (barres)")
    vendor_remain_df = (
        edited_df.groupby("vendor_name", as_index=False)["remaining_amount"].sum()
    )
    st.plotly_chart(
        px.bar(vendor_remain_df, x="vendor_name", y="remaining_amount"),
        use_container_width=True,
    )

    st.subheader("🥧 Répartition du reste à payer")
    st.plotly_chart(
        px.pie(vendor_remain_df, names="vendor_name", values="remaining_amount"),
        use_container_width=True,
    )

    st.subheader("📆 Dépenses par mois")
    month_df = edited_df.groupby("invoice_month", as_index=False)["total_amount"].sum()
    st.plotly_chart(
        px.bar(month_df, x="invoice_month", y="total_amount"),
        use_container_width=True,
    )

    csv_all = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger TOUTES les factures en CSV",
        data=csv_all,
        file_name="invoices_with_payment_tracking.csv",
        mime="text/csv",
    )

    if not unpaid_df.empty:
        csv_unpaid = unpaid_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger UNIQUEMENT les factures non payées (CSV)",
            data=csv_unpaid,
            file_name="invoices_unpaid_only.csv",
            mime="text/csv",
        )


# ============================================================
# 8. UI principale
# ============================================================

def render_main_ui() -> None:
    render_header()

    api_key = get_openai_api_key()
    if not api_key:
        st.error(
            "❌ Aucune clé API OpenAI trouvée.\n"
            "Définis la variable d'environnement `OPENAI_API_KEY`."
        )
        return

    client = build_openai_client(api_key)

    st.subheader("1️⃣ Upload de factures (PDF ou images)")
    uploaded_files = st.file_uploader(
        "Sélectionne une ou plusieurs factures",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🚀 Analyser les factures", type="primary"):
        results: List[Dict[str, Any]] = []

        progress = st.progress(0.0)
        total = len(uploaded_files)

        for idx, f in enumerate(uploaded_files, start=1):
            st.write(f"Analyse de **{f.name}** …")

            invoice_text = get_invoice_text_for_file(client, f)
            if not invoice_text:
                st.warning(f"Impossible de lire le contenu de {f.name}.")
                continue

            structured = structure_invoice_with_gpt(
                client=client,
                invoice_text=invoice_text,
                file_name=f.name,
            )
            results.append(structured)
            progress.progress(idx / total)

        if not results:
            st.error("Aucune facture n'a pu être analysée.")
        else:
            df = invoices_to_dataframe(results)
            if df.empty:
                st.error("Impossible de construire un tableau à partir des factures.")
            else:
                st.session_state["invoices_df"] = df

    if "invoices_df" in st.session_state:
        render_dashboard(st.session_state["invoices_df"])
    else:
        if not uploaded_files:
            st.info("⬆️ Upload au moins une facture pour commencer.")
        else:
            st.info("Clique sur **Analyser les factures** pour générer le dashboard.")


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    configure_page()
    render_main_ui()


if __name__ == "__main__":
    main()

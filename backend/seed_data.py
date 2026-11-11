"""Seed script to populate a verified multilingual benchmark dataset and documents."""

import hashlib
import uuid
from app.db.session import SessionLocal, Base, engine
from app.models.dataset import Dataset, Document, Query, QueryGroup, RelevanceJudgment
from app.models.embedding import DocumentEmbedding
from app.adapters.registry import default_registry

def seed():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    # Check if benchmark dataset exists
    ds = session.query(Dataset).filter_by(slug="multilingual-ir-benchmark").first()
    if not ds:
        ds = Dataset(
            id=uuid.uuid4(),
            name="Global Climate & Agriculture Benchmark",
            slug="multilingual-ir-benchmark",
            version="1.0.0",
            description="Parallel multilingual evaluation benchmark across English, Bengali, Banglish, Hindi, and Arabic.",
            dataset_type="VALIDATED_BENCHMARK",
            license="CC-BY-4.0",
            supported_languages=["en", "bn", "bn-Latn", "hi", "ar"],
            content_hash_sha256=hashlib.sha256(b"multilingual-ir-benchmark-seed").hexdigest(),
            is_locked=True,
        )
        session.add(ds)
        session.commit()
        print(f"Created dataset {ds.name} ({ds.id})")

    # Sample documents across languages
    documents_data = [
        {
            "external_id": "doc-en-agri-01",
            "lang": "en",
            "script": "Latn",
            "title": "Climate Change Impacts on Crop Yields",
            "content": "Rising global temperatures, erratic precipitation, and prolonged droughts significantly reduce agricultural productivity and crop yields across tropical and subtropical farming regions.",
        },
        {
            "external_id": "doc-en-renew-02",
            "lang": "en",
            "script": "Latn",
            "title": "Renewable Energy in Rural Farming",
            "content": "Solar-powered irrigation and wind turbines provide sustainable energy for rural agriculture, reducing carbon emissions and operational farming expenses.",
        },
        {
            "external_id": "doc-bn-agri-01",
            "lang": "bn",
            "script": "Beng",
            "title": "কৃষি ক্ষেত্রে জলবায়ু পরিবর্তনের ক্ষতিকর প্রভাব",
            "content": "ক্রমবর্ধমান বৈশ্বিক তাপমাত্রা, অনিয়মিত বৃষ্টিপাত এবং দীর্ঘস্থায়ী খরা ক্রান্তীয় অঞ্চলের কৃষি উৎপাদন ও ফসলের ফলন ব্যাপকভাবে হ্রাস করছে।",
        },
        {
            "external_id": "doc-bn-renew-02",
            "lang": "bn",
            "script": "Beng",
            "title": "গ্রামীণ কৃষিতে সৌরশক্তি ও নবায়নযোগ্য প্রযুক্তি",
            "content": "সৌরচালিত সেচ পাম্প এবং বায়ুবিদ্যুৎ গ্রামীণ কৃষি খাতে টেকসই জ্বালানি সরবরাহ করে, যা কার্বন নির্গমন এবং কৃষকদের খরচ কমায়।",
        },
        {
            "external_id": "doc-bn-latn-agri-01",
            "lang": "bn-Latn",
            "script": "Latn",
            "title": "Krishi khetre jolobayu poribortoner probhab",
            "content": "Bishshotomapmatra briddhi, oniyomito brishti ebong khora krishi utpadon o fosholer folon prochur porimane komiye dicche.",
        },
        {
            "external_id": "doc-hi-agri-01",
            "lang": "hi",
            "script": "Deva",
            "title": "कृषि पर जलवायु परिवर्तन का प्रतिकूल प्रभाव",
            "content": "बढ़ते वैश्विक तापमान, अनियमित वर्षा और लंबे सूखे के कारण ग्रामीण क्षेत्रों में कृषि उत्पादकता और फसलों की पैदावार में भारी गिरावट आ रही है।",
        },
        {
            "external_id": "doc-ar-agri-01",
            "lang": "ar",
            "script": "Arab",
            "title": "تأثير تغير المناخ على الإنتاجية الزراعية",
            "content": "يؤدي ارتفاع درجات الحرارة العالمية والأمطار غير المنتظمة وفترات الجفاف الطويلة إلى انخفاض حاد في إنتاجية المحاصيل الزراعية في المناطق الريفية.",
        },
    ]

    docs_to_embed = []
    for d in documents_data:
        existing = session.query(Document).filter_by(doc_external_id=d["external_id"]).first()
        if not existing:
            new_doc = Document(
                id=uuid.uuid4(),
                dataset_id=ds.id,
                doc_external_id=d["external_id"],
                language_code=d["lang"],
                script_code=d["script"],
                title=d["title"],
                content=d["content"],
                metadata_json={"domain": "agriculture", "climate_sensitive": True},
            )
            session.add(new_doc)
            docs_to_embed.append(new_doc)
            print(f"Added doc: {d['external_id']} [{d['lang']}]")

    session.commit()

    # Pre-embed documents using both registered models
    for model_meta in default_registry.list_available_models():
        adapter = default_registry.get_adapter(model_meta.model_id)
        all_docs = session.query(Document).filter_by(dataset_id=ds.id).all()
        texts = [doc.content for doc in all_docs]
        print(f"Encoding {len(texts)} documents with {adapter.model_id}...")
        vectors = adapter.encode_documents(texts, normalize=True)

        for doc, vec in zip(all_docs, vectors):
            # Check if embedding already exists
            existing_emb = session.query(DocumentEmbedding).filter_by(
                document_id=doc.id,
                model_id=adapter.model_id
            ).first()
            if not existing_emb:
                emb = DocumentEmbedding(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    model_id=adapter.model_id,
                    dimension=vec.shape[0],
                    embedding=vec.astype("float32").tolist(),
                )
                session.add(emb)
        session.commit()
        print(f"Embeddings saved for {adapter.model_id}.")

    # Update dataset document counter
    ds.document_count = session.query(Document).filter_by(dataset_id=ds.id).count()
    session.commit()
    print("Database seeding completed successfully!")
    session.close()

if __name__ == "__main__":
    seed()

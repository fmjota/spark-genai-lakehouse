"""Tests de la inferencia vectorizada en Spark (mapInPandas) en modo mock."""

from genai_etl.models.inference import enrich_with_nlp, output_schema


def _df_texts(spark, textos):
    return spark.createDataFrame([(str(i), t) for i, t in enumerate(textos)], schema=["id", "text"])


def test_schema_sentimiento_y_ner():
    s = output_schema("sentiment")
    assert "label" in s.fieldNames() and "confidence" in s.fieldNames()
    n = output_schema("ner")
    assert "entities" in n.fieldNames() and "n_entities" in n.fieldNames()


def test_enrich_sentimiento(spark, education_cfg):
    df = _df_texts(spark, ["Excelente curso, muy claro", "Pésimo y desorganizado"])
    out = enrich_with_nlp(df, education_cfg, mode="mock").orderBy("id").collect()
    labels = {r["id"]: r["label"] for r in out}
    assert labels["0"] == "POS"
    assert labels["1"] == "NEG"
    assert all(0.0 <= r["confidence"] <= 1.0 for r in out)


def test_enrich_ner(spark, health_cfg):
    df = _df_texts(spark, ["Tomó ibuprofeno por la cefalea", "Control de rutina sin novedad"])
    out = enrich_with_nlp(df, health_cfg, mode="mock").orderBy("id").collect()
    by_id = {r["id"]: r for r in out}
    tipos = {e["type"] for e in by_id["0"]["entities"]}
    assert {"FARMACO", "SINTOMA"} <= tipos
    assert by_id["0"]["n_entities"] >= 2
    # spans válidos respecto al texto
    e = by_id["0"]["entities"][0]
    assert by_id["0"]["text"][e["start"] : e["end"]] == e["text"]
    # texto sin entidades -> n_entities 0 y confianza 0
    assert by_id["1"]["n_entities"] == 0
    assert by_id["1"]["confidence"] == 0.0


def test_inferencia_es_determinista(spark, banking_cfg):
    df = _df_texts(spark, ["La acción sube fuerte", "Caída y pérdida en el trimestre"])
    r1 = enrich_with_nlp(df, banking_cfg, mode="mock").orderBy("id").collect()
    r2 = enrich_with_nlp(df, banking_cfg, mode="mock").orderBy("id").collect()
    assert [(r["label"], r["confidence"]) for r in r1] == [
        (r["label"], r["confidence"]) for r in r2
    ]

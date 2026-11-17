from fastapi.testclient import TestClient

from app.server import create_app


def test_query_endpoint(tuned_pipeline):
    with TestClient(create_app({"tuned": tuned_pipeline}, default_profile="tuned")) as client:
        resp = client.post("/query", json={"question": "Which bacteria cause most bladder infections?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"][0]["doc_id"] == "uti"
        assert "[1]" in body["answer"] or "[2]" in body["answer"]  # cited answer
        assert body["disclaimer"]


def test_validation_and_unknown_profile(tuned_pipeline):
    with TestClient(create_app({"tuned": tuned_pipeline}, default_profile="tuned")) as client:
        assert client.post("/query", json={"question": ""}).status_code == 422
        assert client.post("/query", json={"question": "hello there", "profile": "nope"}).status_code == 400


def test_health_ready_and_metrics(tuned_pipeline):
    with TestClient(create_app({"tuned": tuned_pipeline}, default_profile="tuned")) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json()["profiles"] == ["tuned"]
        client.post("/query", json={"question": "How is asthma treated?"})
        assert "rag_queries_total" in client.get("/metrics/").text


def test_readyz_fails_without_default_profile(tuned_pipeline):
    with TestClient(create_app({"tuned": tuned_pipeline}, default_profile="baseline")) as client:
        assert client.get("/readyz").status_code == 503

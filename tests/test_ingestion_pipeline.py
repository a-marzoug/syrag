from syrag.observability import ObservabilityHub
from syrag.schemas import IngestRequest
from syrag.services import DefaultIngestionPipeline


def test_default_ingestion_pipeline_builds_source_documents() -> None:
    pipeline = DefaultIngestionPipeline(observability=ObservabilityHub())

    source_documents = pipeline.build_source_documents_for_testing(
        IngestRequest(
            documents=["doc one", "doc two"],
            metadata={"source_id": "doc", "page_number": 1},
        )
    )

    assert source_documents[0].source_id == "doc-0"
    assert source_documents[1].source_id == "doc-1"

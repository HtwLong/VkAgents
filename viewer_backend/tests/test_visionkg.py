from viewer_backend.visionkg import build_class_availability_query, query_class_availability


def test_query_builder_escapes_class_literals():
    query = build_class_availability_query('person" } UNION { ?s ?p ?o')
    assert '"person\\\" } UNION { ?s ?p ?o"' in query
    assert "COUNT(DISTINCT ?image)" in query


def test_availability_parser_and_query_artifact(tmp_path):
    def transport(query):
        assert "cv:hasAnnotation" in query
        return {"results": {"bindings": [
            {"datasetName": {"value": "small"}, "count": {"value": "2"}},
            {"datasetName": {"value": "large"}, "count": {"value": "20"}},
        ]}}

    path = tmp_path / "DATA_CHECK_QUERY.sparql"
    result = query_class_availability(["person"], query_output_path=path, transport=transport)
    assert result == [{"class_name": "person", "sources": [
        {"dataset_name": "large", "count": 20},
        {"dataset_name": "small", "count": 2},
    ]}]
    assert "# Class: person" in path.read_text(encoding="utf-8")

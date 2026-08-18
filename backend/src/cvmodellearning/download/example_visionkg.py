import requests
import pprint
import json
import time
from cvmodellearning.download.download_data import download_visionkg_mixed_datasets_detection, download_visionkg_images_flat
from cvmodellearning.download.visionkg_utils import prepare_data, query as production_query, visionkg2cocoDet
from cvmodellearning.paths import visionkg_cache_dir

def query(query_string, token=""):
      response = requests.get('https://vision.semkg.org/sparql',
                             json={"query": query_string, token: token})
      _data=response.json()
      data=[]
      print("Query Result:")
      pprint.pprint(_data)
      # push test
      for result in _data['results']['bindings']:
            tmp={}
            for key in result.keys():
                  tmp[key]=result[key]['value']
            data.append(tmp)
      return data

def try_download_visionkg():
    query_string='''
    PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
    PREFIX schema:<http://schema.org/>
    PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>

    SELECT ?datasetName (STR(?_imageName) AS ?imageName) (xsd:integer(?_imageWidth) AS ?imageWidth) (xsd:integer(?_imageHeight) AS ?imageHeight) ?labelName ?bbHeight ?bbWidth ?bbCentreX (STR(?imageUrl) AS ?imageUrl)
    WHERE {
    {
        SELECT ?image
        WHERE{
        ?image cv:hasAnnotation ?annotation1.
        ?annotation1 a cv:ObjectDetectionAnnotation.
        ?annotation1 cv:hasLabel ?label1.
        ?label1 cv:label "car".

        }
        GROUP BY ?image
        LIMIT 10
    }
    ?image schema:isPartOf / schema:name ?datasetName .
    ?image schema:name ?_imageName.
    OPTIONAL{?image schema:contentUrl ?imageUrl}.
    ?image cv:hasAnnotation ?annotation.
    ?image cv:imgWidth ?_imageWidth.
    ?image cv:imgHeight ?_imageHeight.
    ?annotation cv:hasLabel/cv:label ?labelName.
    ?annotation cv:hasBox ?bbox.
    ?bbox cv:boxHeight ?bbHeight.
    ?bbox cv:boxWidth ?bbWidth.
    ?bbox cv:centerX ?bbCentreX.
    ?bbox cv:centerY ?bbCenterY.
    }
    '''

    #Query and return result
    result=query(query_string)
    from pprint import pprint
    pprint(result)

def try_download_visionkg_simple():
    query_string='''
    PREFIX cv: <http://vision.semkg.org/onto/v0.1/>
    PREFIX schema: <http://schema.org/>
    
    SELECT ?datasetName ?imageName ?labelName ?bbWidth ?bbHeight ?bbCentreX ?bbCentreY ?imgWidth ?imgHeight
    WHERE {
        {
           # Wrap the UNION in a group
           {
               SELECT ?image ?datasetName ?labelName WHERE {
                    ?image schema:isPartOf / schema:name ?datasetName .
                    FILTER regex(?datasetName, "bdd", "i")
                    ?image cv:hasAnnotation ?annotation .
                    ?annotation cv:hasLabel/cv:label ?labelName .
                    FILTER regex(?labelName, "^car$", "i")
               } LIMIT 1
           }
           
        }
    
        OPTIONAL { ?image schema:name ?imageName } .
        OPTIONAL { ?image cv:imgWidth ?imgWidth . ?image cv:imgHeight ?imgHeight . }
    
        ?image cv:hasAnnotation ?annotation .
        ?annotation cv:hasLabel/cv:label ?labelName .
        FILTER regex(?labelName, "^car$", "i")
    
        OPTIONAL {
            ?annotation cv:hasBox ?bbox .
            OPTIONAL { ?bbox cv:boxWidth ?bbWidth } .
            OPTIONAL { ?bbox cv:boxHeight ?bbHeight } .
            OPTIONAL { ?bbox cv:centerX ?bbCentreX . ?bbox cv:centerY ?bbCentreY } .
        }
    }
    '''

    #Query and return result
    result=query(query_string)
    from pprint import pprint
    pprint(result)


def try_download_visionkg_simple2():
    query_string='''PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
    PREFIX schema:<http://schema.org/>

SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image  ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
WHERE
{
    {
        {
            SELECT DISTINCT ?image  {
                ?image schema:isPartOf / schema:name "bdd_100k_det_train" .
                ?image cv:hasAnnotation ?annotation1 .
                ?annotation1 a cv:ObjectDetectionAnnotation.
                ?annotation1 cv:hasLabel ?label1 .
                ?label1 cv:label "pedestrian" .
            }
            LIMIT 10
        }

    }

    ?image cv:imgWidth ?imageWidth .
    ?image cv:imgHeight ?imageHeight .

    ?image schema:isPartOf / schema:name ?datasetName .
    ?image schema:name ?imageName .

    ?image cv:hasAnnotation ?annotation .
    ?annotation cv:hasLabel/cv:label ?labelName .
    OPTIONAL{?annotation cv:hasBox ?bbox} .
    ?bbox cv:boxHeight ?bbHeight .
    ?bbox cv:boxWidth ?bbWidth .
    ?bbox cv:centerX ?bbCentreX .
    ?bbox cv:centerY ?bbCentreY .

}
    '''

    #Query and return result
    result=query(query_string)
    from pprint import pprint
    pprint(result)

def try_download_visionkg_simple3():
    query_string='''
    
    PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
    PREFIX schema:<http://schema.org/>
    PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
    
    
    SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image ?imageUrl 
           ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
    WHERE {
        
        # --- INNER UNION BLOCK ---
        {
            
            
            {
                SELECT DISTINCT ?image ?imageUrl
                WHERE {
                    ?image schema:isPartOf / schema:name "coco2017_det_train" .
                    ?image schema:contentUrl ?imageUrl .

                    ?image cv:hasAnnotation ?ann .
                    ?ann a cv:ObjectDetectionAnnotation .
                    ?ann cv:hasLabel ?lbl .
                    ?lbl cv:label "bicycle" .
                }
                LIMIT 30
            }
             UNION 
            {
                SELECT DISTINCT ?image ?imageUrl
                WHERE {
                    ?image schema:isPartOf / schema:name "bdd_100k_det_train" .
                    ?image schema:contentUrl ?imageUrl .

                    ?image cv:hasAnnotation ?ann .
                    ?ann a cv:ObjectDetectionAnnotation .
                    ?ann cv:hasLabel ?lbl .
                    ?lbl cv:label "bicycle" .
                }
                LIMIT 20
            }
             UNION 
            {
                SELECT DISTINCT ?image ?imageUrl
                WHERE {
                    ?image schema:isPartOf / schema:name "bdd_100k_det_train" .
                    ?image schema:contentUrl ?imageUrl .

                    ?image cv:hasAnnotation ?ann .
                    ?ann a cv:ObjectDetectionAnnotation .
                    ?ann cv:hasLabel ?lbl .
                    ?lbl cv:label "pedestrian" .
                }
                LIMIT 100
            }
             UNION 
            {
                SELECT DISTINCT ?image ?imageUrl
                WHERE {
                    ?image schema:isPartOf / schema:name "KITTI_det" .
                    ?image schema:contentUrl ?imageUrl .

                    ?image cv:hasAnnotation ?ann .
                    ?ann a cv:ObjectDetectionAnnotation .
                    ?ann cv:hasLabel ?lbl .
                    ?lbl cv:label "pedestrian" .
                }
                LIMIT 30
            }
            
        }
        
        # --- DATA FETCHING (Outer Join) ---
        ?image cv:imgWidth ?imageWidth .
        ?image cv:imgHeight ?imageHeight .
        ?image schema:isPartOf / schema:name ?datasetName .
        ?image schema:name ?imageName .

        ?image cv:hasAnnotation ?annotation .
        ?annotation cv:hasLabel/cv:label ?labelName .
        
        OPTIONAL {?annotation cv:hasBox ?bbox }.
        ?bbox cv:boxHeight ?bbHeight .
        ?bbox cv:boxWidth ?bbWidth .
        ?bbox cv:centerX ?bbCentreX .
        ?bbox cv:centerY ?bbCentreY .
        
    }
    
    '''

    #Query and return result
    result=query(query_string)
    from pprint import pprint
    pprint(result)


def try_download_visionkg_simple4():
    query_string = f"""
            PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
            PREFIX schema:<http://schema.org/>
            PREFIX xsd:<http://www.w3.org/2001/XMLSchema#>
            
            SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image 
                   ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
            WHERE {{
                # --- INNER SUBQUERY (Limits by Image Count) ---
                {{
                    SELECT DISTINCT ?image ?datasetName 
                    WHERE {{
                        ?image schema:isPartOf / schema:name ?datasetName .
                        FILTER regex(?datasetName, "bdd_100k_det_train", "i")
                        
                        ?image cv:hasAnnotation ?ann .
                        ?ann a cv:ObjectDetectionAnnotation .
                        ?ann cv:hasLabel/cv:label ?labelName .
                        FILTER regex(?labelName, "bicycle", "i")
                    }}
                    LIMIT 1
                }}

                # --- OUTER DATA FETCHING ---
                OPTIONAL {{ ?image schema:name ?imageName }} .
                OPTIONAL {{ 
                    ?image cv:imgWidth ?imageWidth .
                    ?image cv:imgHeight ?imageHeight .
                }}

                ?image cv:hasAnnotation ?annotation .
                ?annotation cv:hasLabel/cv:label ?labelName .
                
                # We filter again in the outer scope to ensure we only get boxes 
                # for the specific class we asked for (in case the image has multiple classes)
                FILTER regex(?labelName, "bicycle", "i")
                
                OPTIONAL {{
                    ?annotation cv:hasBox ?bbox .
                    OPTIONAL {{?bbox cv:boxHeight ?bbHeight }} .
                    OPTIONAL {{?bbox cv:boxWidth ?bbWidth }} .
                    OPTIONAL {{?bbox cv:centerX ?bbCentreX . ?bbox cv:centerY ?bbCentreY }} .
                }}
            }}
            """

    #Query and return result
    result=query(query_string)
    from pprint import pprint
    pprint(result)

def inspect_selected_config_flat_urls():
    """Print one image URL per selected class/dataset pair without downloading."""
    selected_data = [
        {
            "class_name": "car",
            "sources": [
                {"dataset_name": "bdd_100k_det_train", "count": 1},
                {"dataset_name": "openimages_challenge_2019_det_train", "count": 1},
                {"dataset_name": "bdd100k_UNIT_day2night_det_train", "count": 1},
                {"dataset_name": "coco2017_det_train", "count": 1},
                {"dataset_name": "ACDC_det_val_night", "count": 1},
            ],
        },
        {
            "class_name": "truck",
            "sources": [
                {"dataset_name": "objects365_det_train", "count": 1},
                {"dataset_name": "bdd_100k_det_train", "count": 1},
                {"dataset_name": "openimages_challenge_2019_det_train", "count": 1},
                {"dataset_name": "bdd100k_cycleGAN_day2night_det_train", "count": 1},
            ],
        },
        {
            "class_name": "bus",
            "sources": [
                {"dataset_name": "objects365_det_train", "count": 1},
                {"dataset_name": "bdd_100k_det_train", "count": 1},
                {"dataset_name": "coco2017_det_train", "count": 1},
            ],
        },
        {
            "class_name": "motorcycle",
            "sources": [
                {"dataset_name": "objects365_det_train", "count": 1},
                {"dataset_name": "openimages_challenge_2019_det_train", "count": 1},
                {"dataset_name": "coco2017_det_train", "count": 1},
                {"dataset_name": "LVIS_det_train", "count": 1},
                {"dataset_name": "bdd100k_UNIT_day2night_det_train", "count": 1},
            ],
        },
    ]
    download_visionkg_images_flat(
        "selected_config_flat_example",
        selected_data,
        download=False,
    )


def try_download_person_10_with_production_query(
    dataset_name="openimages_challenge_2019_det_train",
    split="train",
):
    """Download ten Open Images person samples through the production code path.

    This deliberately uses the same query construction, response validation,
    annotation conversion, persistent image cache, and run materialization as
    the detection download endpoint. Only the requested allocation is smaller.
    """

    selected_data = [{
        "class_name": "person",
        "sources": [{
            "dataset_name": dataset_name,
            "allocations": [{
                "split": split,
                "count": 10,
                "assignment_type": "official_split",
            }],
        }],
    }]
    report = download_visionkg_mixed_datasets_detection(
        f"visionkg-example-person-10-{split}",
        selected_data,
    )
    print(json.dumps(report, indent=2))
    return report


def try_download_person_10_with_exact_bindings():
    """Test an alternative query locally without changing the pipeline query."""

    query_string = """
    PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
    PREFIX schema:<http://schema.org/>

    SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image
           ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
    WHERE {
        {
            SELECT DISTINCT ?image ?datasetName
            WHERE {
                VALUES ?datasetName { "openimages_challenge_2019_det_val" }
                ?image schema:isPartOf / schema:name ?datasetName .
                ?image cv:hasAnnotation ?ann .
                ?ann cv:hasLabel/cv:label "person" .
            }
            ORDER BY STR(?image)
            LIMIT 10
            OFFSET 0
        }

        ?image schema:name ?imageName .
        ?image cv:imgWidth ?imageWidth .
        ?image cv:imgHeight ?imageHeight .
        ?image cv:hasAnnotation ?annotation .
        ?annotation cv:hasLabel/cv:label ?labelName .
        VALUES ?labelName { "person" }
        ?annotation cv:hasBox ?bbox .
        ?bbox cv:boxHeight ?bbHeight .
        ?bbox cv:boxWidth ?bbWidth .
        ?bbox cv:centerX ?bbCentreX .
        ?bbox cv:centerY ?bbCentreY .
    }
    """
    print("SPARQL query:\n", query_string.strip())
    rows = production_query(query_string)
    coco = visionkg2cocoDet(rows)
    result = prepare_data(coco["images"], DATA_ROOT_PATH=str(visionkg_cache_dir()))
    summary = {
        "query_rows": len(rows),
        "candidate_images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "successful_downloads": len(result["successful"]),
        "failed_downloads": len(result["failures"]),
        "metrics": result["metrics"],
    }
    print(json.dumps(summary, indent=2))
    return summary


def compare_person_query_variants(limits=(10, 100, 500, 1400)):
    """Benchmark candidate-filter variants without changing production code."""

    dataset_name = "openimages_challenge_2019_det_val"
    class_name = "person"
    variants = {
        "values": (
            f'VALUES ?datasetName {{ "{dataset_name}" }}\n'
            "                ?image schema:isPartOf / schema:name ?datasetName .\n"
            "                ?image cv:hasAnnotation ?ann .\n"
            f'                ?ann cv:hasLabel/cv:label "{class_name}" .'
        ),
        "str_equality": (
            "?image schema:isPartOf / schema:name ?datasetName .\n"
            f'                FILTER(STR(?datasetName) = "{dataset_name}")\n'
            "                ?image cv:hasAnnotation ?ann .\n"
            "                ?ann cv:hasLabel/cv:label ?candidateLabel .\n"
            f'                FILTER(STR(?candidateLabel) = "{class_name}")'
        ),
        "lcase_str": (
            "?image schema:isPartOf / schema:name ?datasetName .\n"
            f'                FILTER(LCASE(STR(?datasetName)) = LCASE("{dataset_name}"))\n'
            "                ?image cv:hasAnnotation ?ann .\n"
            "                ?ann cv:hasLabel/cv:label ?candidateLabel .\n"
            f'                FILTER(LCASE(STR(?candidateLabel)) = LCASE("{class_name}"))'
        ),
        "regex": (
            "?image schema:isPartOf / schema:name ?datasetName .\n"
            f'                FILTER regex(?datasetName, "^{dataset_name}$", "i")\n'
            "                ?image cv:hasAnnotation ?ann .\n"
            "                ?ann cv:hasLabel/cv:label ?candidateLabel .\n"
            f'                FILTER regex(?candidateLabel, "^{class_name}$", "i")'
        ),
    }
    results = []
    for variant_name, candidate_pattern in variants.items():
        for ordered in (False, True):
            for limit in limits:
                order_clause = "ORDER BY STR(?image)" if ordered else ""
                query_string = f"""
                PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
                PREFIX schema:<http://schema.org/>

                SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image
                       ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
                WHERE {{
                    {{
                        SELECT DISTINCT ?image ?datasetName
                        WHERE {{
                            {candidate_pattern}
                        }}
                        {order_clause}
                        LIMIT {limit}
                        OFFSET 0
                    }}
                    ?image schema:name ?imageName .
                    ?image cv:imgWidth ?imageWidth .
                    ?image cv:imgHeight ?imageHeight .
                    ?image cv:hasAnnotation ?annotation .
                    ?annotation cv:hasLabel/cv:label ?labelName .
                    VALUES ?labelName {{ "person" }}
                    ?annotation cv:hasBox ?bbox .
                    ?bbox cv:boxHeight ?bbHeight .
                    ?bbox cv:boxWidth ?bbWidth .
                    ?bbox cv:centerX ?bbCentreX .
                    ?bbox cv:centerY ?bbCentreY .
                }}
                """
                started = time.perf_counter()
                try:
                    rows = production_query(query_string)
                    record = {
                        "variant": variant_name,
                        "ordered": ordered,
                        "limit": limit,
                        "status": "ok",
                        "rows": len(rows),
                        "images": len({row.get("image") for row in rows}),
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                except Exception as exc:
                    record = {
                        "variant": variant_name,
                        "ordered": ordered,
                        "limit": limit,
                        "status": "error",
                        "error": str(exc),
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                results.append(record)
                print(json.dumps(record))
                if record["status"] == "error":
                    break
    return results


def test_recommended_multilabel_query(limit=1400, offset=1400):
    """Exercise the proposed production query with multi-label boxes and paging."""

    query_string = f"""
    PREFIX cv:<http://vision.semkg.org/onto/v0.1/>
    PREFIX schema:<http://schema.org/>

    SELECT ?datasetName ?imageWidth ?imageHeight ?imageName ?image
           ?labelName ?bbHeight ?bbWidth ?bbCentreX ?bbCentreY
    WHERE {{
        {{
            SELECT DISTINCT ?image ?datasetName
            WHERE {{
                VALUES ?datasetName {{ "openimages_challenge_2019_det_val" }}
                ?image schema:isPartOf / schema:name ?datasetName .
                ?image cv:hasAnnotation ?ann .
                ?ann cv:hasLabel/cv:label "person" .
            }}
            ORDER BY STR(?image)
            LIMIT {limit}
            OFFSET {offset}
        }}
        ?image schema:name ?imageName .
        ?image cv:imgWidth ?imageWidth .
        ?image cv:imgHeight ?imageHeight .
        ?image cv:hasAnnotation ?annotation .
        ?annotation cv:hasLabel/cv:label ?labelName .
        VALUES ?labelName {{ "cat" "dog" "person" }}
        ?annotation cv:hasBox ?bbox .
        ?bbox cv:boxHeight ?bbHeight .
        ?bbox cv:boxWidth ?bbWidth .
        ?bbox cv:centerX ?bbCentreX .
        ?bbox cv:centerY ?bbCentreY .
    }}
    """
    started = time.perf_counter()
    rows = production_query(query_string)
    result = {
        "limit": limit,
        "offset": offset,
        "rows": len(rows),
        "images": len({row.get("image") for row in rows}),
        "labels": sorted({row.get("labelName") for row in rows}),
        "seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    test_recommended_multilabel_query()

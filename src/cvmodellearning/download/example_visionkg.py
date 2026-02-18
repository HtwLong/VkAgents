import requests
import pprint
from cvmodellearning.download.download_data import download_visionkg_mixed_datasets

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

#try_download_visionkg_simple4()
download_visionkg_mixed_datasets("experiment_001",  [ { "class_name": "bicycle", "sources": [ { "dataset_name": "objects365_det_train", "image_count": 1 }, { "dataset_name": "coco2017_det_train", "image_count": 1 }, { "dataset_name": "bdd_100k_det_train", "image_count": 1 } ] }, { "class_name": "pedestrian", "sources": [ { "dataset_name": "bdd_100k_det_train", "image_count": 1 }, { "dataset_name": "KITTI_det", "image_count": 1} ] } ])


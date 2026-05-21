# =========================================
# IMPORT ELASTICSEARCH
# =========================================

from elasticsearch import Elasticsearch

# =========================================
# CONNECT TO ELASTICSEARCH
# =========================================

es = Elasticsearch(

    "http://localhost:9200"
)

# =========================================
# INDEX ALERT FUNCTION
# =========================================

def index_alert(alert_data):

    response = es.index(

        index="ai_alerts",

        document=alert_data
    )

    return response
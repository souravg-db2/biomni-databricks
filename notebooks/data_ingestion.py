# Databricks notebook source
# MAGIC %md
# MAGIC # Biomni Data Ingestion to UC Volume
# MAGIC
# MAGIC Downloads data lake files from S3 and writes them to `/Volumes/biomni/agent/raw_files`.
# MAGIC Run once (or on a schedule) to populate the volume. The A1 agent reads from this path.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ensure catalog, schema, and volume exist

# COMMAND ----------

# Create catalog and schema if they do not exist (requires privileges)
# spark.sql("CREATE CATALOG IF NOT EXISTS biomni")
# spark.sql("CREATE SCHEMA IF NOT EXISTS biomni.agent")
# spark.sql("CREATE VOLUME IF NOT EXISTS biomni.agent.raw_files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Download data lake files from S3 and write to volume

# COMMAND ----------

import os
import tempfile
from urllib.parse import urljoin

# Volume path (must exist). Use FUSE path if available, else we write to local then dbutils.fs.cp
VOLUME_PATH = "/Volumes/biomni/agent/raw_files"
S3_BUCKET = "https://biomni-release.s3.amazonaws.com"
FOLDER = "data_lake"

# List of expected files (from env_desc.data_lake_dict)
EXPECTED_FILES = [
    "affinity_capture-ms.parquet",
    "affinity_capture-rna.parquet",
    "BindingDB_All_202409.tsv",
    "gene_info.parquet",
    "gwas_catalog.pkl",
    "marker_celltype.parquet",
    "sgRNA_KO_SP_mouse.txt",
    "sgRNA_KO_SP_human.txt",
    # Add more from biomni_agent.env_desc.data_lake_dict as needed
]

# COMMAND ----------

def download_file(url: str, dest_path: str) -> bool:
    import requests
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# COMMAND ----------

# Download each file to a temp dir, then copy to volume (Databricks FUSE or dbutils)
try:
    dbutils = dbutils  # noqa: F821
except NameError:
    dbutils = None

for filename in EXPECTED_FILES:
    url = urljoin(S3_BUCKET + "/" + FOLDER + "/", filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.basename(filename)) as tmp:
        tmp_path = tmp.name
    if download_file(url, tmp_path):
        dest = f"{VOLUME_PATH.rstrip('/')}/{filename}"
        try:
            if dbutils:
                with open(tmp_path, "rb") as f:
                    dbutils.fs.put(dest, f.read(), overwrite=True)
            else:
                os.makedirs(VOLUME_PATH, exist_ok=True)
                import shutil
                shutil.copy(tmp_path, os.path.join(VOLUME_PATH, filename))
        except Exception as e:
            print(f"Could not write to volume {dest}: {e}")
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify

# COMMAND ----------

if dbutils:
    display(dbutils.fs.ls(VOLUME_PATH))

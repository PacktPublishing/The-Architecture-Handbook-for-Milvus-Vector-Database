{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "15773612",
   "metadata": {},
   "source": [
    "# Chapter 6 – Milvus Delete & Upsert Operations\n",
    "This notebook demonstrates inserting, deleting, and upserting data in a Milvus collection using the `MongoDB/embedded_movies` dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c08c07e3",
   "metadata": {},
   "source": [
    "## Setup: Schema & Collection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "13030516",
   "metadata": {},
   "outputs": [],
   "source": [
    "import itertools\n",
    "from datasets import load_dataset\n",
    "from pymilvus import MilvusClient, DataType\n",
    "\n",
    "client = MilvusClient(uri=\"http://localhost:19530\")\n",
    "\n",
    "schema = MilvusClient.create_schema()\n",
    "schema.add_field(field_name=\"id\", datatype=DataType.INT64, is_primary=True)\n",
    "schema.add_field(field_name=\"title\", datatype=DataType.VARCHAR, max_length=512)\n",
    "schema.add_field(field_name=\"plot\", datatype=DataType.VARCHAR, max_length=65535)\n",
    "schema.add_field(field_name=\"genre\", datatype=DataType.VARCHAR, max_length=64)\n",
    "schema.add_field(field_name=\"embedding\", datatype=DataType.FLOAT_VECTOR, dim=1536)\n",
    "\n",
    "index = MilvusClient.prepare_index_params(field_name=\"embedding\", index_type=\"AUTOINDEX\")\n",
    "client.create_collection(collection_name=\"ch6_movies\", schema=schema, index_params=index)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51d87d8c",
   "metadata": {},
   "source": [
    "## Load & Insert Data\n",
    "Stream the first 500 records from the `MongoDB/embedded_movies` dataset and insert those with a valid plot embedding and genre."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cba8804f-ce14-4cd3-8a08-afc52b66cfa7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'collection_name': 'ch6_movies',\n",
       " 'auto_id': False,\n",
       " 'num_shards': 1,\n",
       " 'description': '',\n",
       " 'fields': [{'field_id': 100,\n",
       "   'name': 'id',\n",
       "   'description': '',\n",
       "   'type': <DataType.INT64: 5>,\n",
       "   'params': {},\n",
       "   'is_primary': True},\n",
       "  {'field_id': 101,\n",
       "   'name': 'title',\n",
       "   'description': '',\n",
       "   'type': <DataType.VARCHAR: 21>,\n",
       "   'params': {'max_length': 512}},\n",
       "  {'field_id': 102,\n",
       "   'name': 'plot',\n",
       "   'description': '',\n",
       "   'type': <DataType.VARCHAR: 21>,\n",
       "   'params': {'max_length': 65535}},\n",
       "  {'field_id': 103,\n",
       "   'name': 'genre',\n",
       "   'description': '',\n",
       "   'type': <DataType.VARCHAR: 21>,\n",
       "   'params': {'max_length': 64}},\n",
       "  {'field_id': 104,\n",
       "   'name': 'embedding',\n",
       "   'description': '',\n",
       "   'type': <DataType.FLOAT_VECTOR: 101>,\n",
       "   'params': {'dim': 1536}}],\n",
       " 'functions': [],\n",
       " 'aliases': [],\n",
       " 'collection_id': 464641688225974855,\n",
       " 'consistency_level': 2,\n",
       " 'properties': {},\n",
       " 'num_partitions': 1,\n",
       " 'enable_dynamic_field': False,\n",
       " 'created_timestamp': 464641820272885765,\n",
       " 'update_timestamp': 464641820272885765}"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "client.describe_collection(\"ch6_movies\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "43381c63",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Insert result: {'insert_count': 493, 'ids': [4465, 10146, 16634, 16654, 16895, 21140, 22792, 23427, 23551, 26205, 27438, 27623, 29030, 29047, 29843, 32762, 32850, 34428, 34522, 34928, 35153, 35530, 35616, 29843, 40724, 40740, 40835, 41163, 41841, 42553, 42646, 42744, 44072, 44324, 45125, 45679, 45966, 46011, 46534, 47073, 48667, 48966, 49101, 49223, 49279, 50356, 50490, 50762, 50858, 51337, 52151, 52365, 52415, 53374, 53804, 54310, 54757, 54953, 55928, 56059, 56197, 57076, 57193, 58150, 58203, 58461, 58525, 59095, 59243, 59549, 59800, 59915, 60490, 60588, 60635, 60862, 60980, 61189, 61369, 61647, 61695, 61933, 62512, 62765, 64045, 64072, 64256, 64387, 64757, 64866, 65051, 65207, 65214, 65570, 65579, 65720, 66064, 66079, 66078, 66372, 66549, 66832, 66920, 66995, 66999, 67116, 67397, 67741, 68638, 68718, 68767, 68935, 69095, 69113, 69332, 69713, 69697, 69768, 69865, 70034, 70328, 70726, 70909, 70947, 71402, 71455, 71521, 71569, 71566, 71807, 72012, 72251, 72281, 72308, 72705, 72737, 72860, 72886, 72901, 72912, 73018, 73282, 73349, 73631, 73707, 73906, 74205, 74442, 74812, 74836, 74857, 74962, 75223, 75627, 75669, 75683, 75765, 76544, 76716, 76729, 76752, 76725, 76759, 76788, 76804, 76993, 77294, 77278, 77369, 77451, 77523, 77617, 77696, 77864, 78346, 78435, 78492, 78490, 78740, 78856, 78869, 78975, 79351, 79501, 79574, 79859, 79891, 79946, 80421, 80455, 80472, 80513, 80684, 80736, 80745, 80801, 80907, 80934, 80997, 81400, 81568, 81573, 81609, 82136, 82250, 82288, 82340, 82356, 82398, 82525, 82648, 82694, 82699, 82869, 82949, 82971, 83190, 83284, 83366, 83511, 83630, 83739, 83944, 84266, 84316, 84671, 84726, 84749, 84756, 84827, 84887, 84935, 85127, 85211, 85255, 85318, 85333, 85811, 85935, 86006, 86034, 86058, 86074, 86190, 86345, 86379, 86393, 86443, 86605, 86606, 86859, 86823, 86896, 86960, 86993, 87032, 87062, 87065, 87078, 87182, 87262, 87344, 87469, 87538, 87597, 87578, 87777, 88011, 87985, 88024, 88044, 88170, 88206, 88194, 88224, 88247, 88925, 88944, 89087, 89092, 89118, 89177, 89243, 89283, 89371, 89378, 89374, 89421, 89461, 89489, 89530, 89607, 89691, 89861, 89881, 89880, 89893, 89901, 89941, 90022, 90180, 90213, 90217, 90264, 90350, 90568, 90605, 90693, 90735, 90728, 90753, 90779, 90859, 90887, 90927, 90915, 90952, 91069, 91060, 91091, 91129, 91187, 91225, 91326, 91344, 91428, 91427, 91431, 91499, 91607, 91637, 91724, 91818, 91875, 91981, 92038, 92106, 92099, 92263, 92493, 92501, 92513, 92534, 92644, 92641, 92675, 92746, 92751, 93015, 93011, 93120, 93185, 93229, 93260, 93278, 93405, 93409, 93428, 93435, 93507, 93560, 93578, 93668, 93692, 93773, 93780, 93771, 93870, 93894, 94074, 94612, 94625, 94631, 94651, 94792, 94894, 94961, 95016, 95250, 95366, 95382, 95403, 95441, 95631, 95655, 95709, 95863, 95866, 95956, 95977, 95981, 96161, 96193, 96256, 96310, 96425, 96446, 96465, 96487, 96535, 96874, 96895, 96913, 96933, 97202, 97244, 97444, 97499, 97576, 97647, 97662, 97733, 97770, 97742, 97818, 97880, 97889, 97910, 97967, 98019, 98155, 98165, 98188, 98206, 98222, 98360, 98369, 98382, 98439, 98467, 98471, 98622, 98691, 98967, 98987, 99005, 99026, 99088, 99160, 99197, 99277, 99327, 99371, 99365, 99422, 99423, 99558, 99652, 99701, 99740, 99753, 99810, 99938, 99991, 100133, 100224, 100263, 100261, 100403, 100502, 100758, 100802, 100912, 100963, 101020, 101393, 101453, 101458, 101764, 102005, 101988, 102004, 102059, 102070, 102095, 102266, 102526, 102593, 102685, 102744, 102798, 102800, 102803, 102820, 102848, 102975, 102984, 103060, 103064, 103045, 103112, 103184, 103285]}\n"
     ]
    }
   ],
   "source": [
    "dataset = load_dataset(\"MongoDB/embedded_movies\", split=\"train\", streaming=True)\n",
    "\n",
    "rows = []\n",
    "for row in itertools.islice(dataset, 500):\n",
    "    if row.get(\"plot_embedding\") and row.get(\"genres\"):\n",
    "        rows.append({\n",
    "            \"id\": row[\"imdb\"][\"id\"],\n",
    "            \"title\": row[\"title\"],\n",
    "            \"plot\": row.get(\"plot\", \"\"),\n",
    "            \"genre\": row[\"genres\"][0],\n",
    "            \"embedding\": row[\"plot_embedding\"],\n",
    "        })\n",
    "\n",
    "insert_res = client.insert(collection_name=\"ch6_movies\", data=rows)\n",
    "print(f\"Insert result: {insert_res}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ba92847",
   "metadata": {},
   "source": [
    "## Delete Operations\n",
    "Milvus deletes are **synchronous at the WAL level** — the Proxy writes the delete record to the message queue and returns immediately. Background DataNodes then apply the deletes to persistent storage asynchronously."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9329e03",
   "metadata": {},
   "source": [
    "### 1. Delete by Primary Key"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0211eaca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Deleting movie with ID: 4465\n",
      "Delete by PK result: {'delete_count': 1}\n"
     ]
    }
   ],
   "source": [
    "ids_to_delete = rows[0]['id']\n",
    "print(f\"Deleting movie with ID: {ids_to_delete}\")\n",
    "res_pk = client.delete(collection_name=\"ch6_movies\", ids=[ids_to_delete])\n",
    "print(f\"Delete by PK result: {res_pk}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a500212",
   "metadata": {},
   "source": [
    "### 2. Delete by Filter Expression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8445e859",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Deleting movies with filter: id <= 4465\n",
      "Delete by filter result: {}\n"
     ]
    }
   ],
   "source": [
    "filter_expr = f'id <= {ids_to_delete}'\n",
    "print(f\"Deleting movies with filter: {filter_expr}\")\n",
    "res_filter = client.delete(collection_name=\"ch6_movies\", filter=filter_expr)\n",
    "print(f\"Delete by filter result: {res_filter}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab0f605f",
   "metadata": {},
   "source": [
    "### 3. Verify Deletion (Asynchronous Phase)\n",
    "The delete is durable in the message queue at this point. The query below should return an empty result."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d80d53c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "res = client.query(\n",
    "    collection_name=\"ch6_movies\",\n",
    "    filter=f'id == \"{ids_to_delete}\"',\n",
    "    output_fields=[\"id\", \"title\"]\n",
    ")\n",
    "print(f\"Query result after delete (should be empty): {res}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b9f749c",
   "metadata": {},
   "source": [
    "## Upsert Example\n",
    "Upsert updates an existing entity if the primary key exists, or inserts it as new otherwise."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "606e7f01",
   "metadata": {},
   "outputs": [],
   "source": [
    "upsert_entities = []\n",
    "\n",
    "# Update existing entity (row 2)\n",
    "upsert_entities.append({\n",
    "    \"id\": rows[2]['id'],\n",
    "    \"title\": \"Updated Title\",\n",
    "    \"plot\": \"Updated plot content\",\n",
    "    \"genre\": rows[2]['genre'],\n",
    "    \"embedding\": rows[2]['embedding']\n",
    "})\n",
    "\n",
    "# Insert new entities (rows 4–5)\n",
    "for i in range(4, min(6, len(rows))):\n",
    "    upsert_entities.append({\n",
    "        \"id\": rows[i]['id'],\n",
    "        \"title\": rows[i]['title'],\n",
    "        \"plot\": rows[i]['plot'],\n",
    "        \"genre\": rows[i]['genre'],\n",
    "        \"embedding\": rows[i]['embedding']\n",
    "    })\n",
    "\n",
    "upsert_res = client.upsert(collection_name=\"ch6_movies\", data=upsert_entities)\n",
    "print(f\"Upsert result: {upsert_res}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

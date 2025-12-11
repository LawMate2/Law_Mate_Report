from typing import List, Tuple, Dict
import psycopg2
from psycopg2.extras import execute_values
from . import VectorStore


class PgVectorStore(VectorStore):
    """PostgreSQL with pgvector extension wrapper"""

    def __init__(self, table_name: str = "rag_benchmark", dimension: int = 768,
                 host: str = "localhost", port: int = 5432,
                 database: str = "vectordb", user: str = "postgres", password: str = "postgres"):
        super().__init__("pgvector", dimension)

        self.table_name = table_name

        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

        # Enable pgvector extension
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Drop table if exists
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

        # Create table
        self.cursor.execute(f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                text TEXT,
                embedding vector({dimension})
            );
        """)

        # Create index for vector similarity search
        self.cursor.execute(f"""
            CREATE INDEX ON {table_name} USING ivfflat (embedding vector_l2_ops)
            WITH (lists = 100);
        """)

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
        """Add texts with embeddings to pgvector"""
        data = [(text, str(embedding)) for text, embedding in zip(texts, embeddings)]

        execute_values(
            self.cursor,
            f"INSERT INTO {self.table_name} (text, embedding) VALUES %s",
            data,
            template="(%s, %s::vector)"
        )

        self.num_vectors += len(texts)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors using L2 distance"""
        query_vector = str(query_embedding)

        self.cursor.execute(f"""
            SELECT text, embedding <-> %s::vector AS distance
            FROM {self.table_name}
            ORDER BY distance
            LIMIT %s;
        """, (query_vector, k))

        results = self.cursor.fetchall()
        return [(text, float(distance)) for text, distance in results]

    def delete_collection(self):
        """Delete the table"""
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name};")
        except:
            pass
        finally:
            try:
                self.cursor.close()
                self.conn.close()
            except:
                pass
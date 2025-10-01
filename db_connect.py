

# from neo4j import GraphDatabase, exceptions, basic_auth
#
# uri = "bolt://44.203.117.86:7687"
# user = "neo4j"
# password = "lasers-excesses-contribution"
# driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
#
# try:
#     with driver.session() as session:
#         # Simple test query
#         result = session.run("RETURN 1 AS test")
#         for record in result:
#             print("Connected! Test query returned:", record["test"])
# except exceptions.ServiceUnavailable as e:
#     print("Connection failed:", e)
# cypher = """
# MERGE (p:Person {name: $name})
# MERGE (pi:PersonInfo {email: $email})
# MERGE (p)-[:HAS_INFO]->(pi)
# RETURN p, pi
# """
#
# with driver.session() as session:
#     result = session.run(cypher, name="Kire", email="kire@example.com")
#     for record in result:
#         print(record["p"])
#         print(record["pi"])

# cypher = """
# MATCH (p:Person)-[r:HAS_INFO]->(pi:PersonInfo)
# RETURN p.name AS person_name, pi.email AS info_email
# """
#
# with driver.session() as session:
#     result = session.run(cypher)
#     for record in result:
#         print(f"Person: {record['person_name']}, Email: {record['info_email']}")

from neo4j import GraphDatabase, exceptions, basic_auth

class Neo4jSandbox:
    def __init__(self, uri, user, password):
        """Initialize connection to Neo4j Sandbox"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the connection"""
        self.driver.close()

    def set_property(self, person_name, property_name, property_value):
        """
        Set a new property or update an existing property on PersonInfo node
        connected to the given Person.
        """
        cypher = f"""
        MATCH (p:Person {{name: $person_name}})-[:HAS_INFO]->(pi:PersonInfo)
        SET pi.{property_name} = $property_value
        RETURN pi
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, person_name=person_name, property_value=property_value)
                for record in result:
                    print(f"Updated PersonInfo property '{property_name}':", record["pi"])
        except exceptions.ServiceUnavailable as e:
            print("Connection failed:", e)
    def add_info_to_person(self, person_name, text_to_add):
        """
        Add or append text to the 'info' property of the PersonInfo node
        connected to a Person node with the given name.
        """
        cypher = """
        MERGE (p:Person {name: $person_name})
        MERGE (pi:PersonInfo)
        MERGE (p)-[:HAS_INFO]->(pi)
        SET pi.info = coalesce(pi.info, '') + $text_to_add
        RETURN p, pi
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, person_name=person_name, text_to_add=text_to_add)
                for record in result:
                    print("Person:", record["p"])
                    print("PersonInfo:", record["pi"])
        except exceptions.ServiceUnavailable as e:
            print("Connection failed:", e)

    def get_info(self, person_name):
        """
        Retrieve the 'info' property from the PersonInfo node
        connected to the given Person.
        Returns '' if not found.
        """
        cypher = """
        MATCH (p:Person {name: $person_name})-[:HAS_INFO]->(pi:PersonInfo)
        RETURN coalesce(pi.Info, '') AS info
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, person_name=person_name)
                record = result.single()
                if record:
                    return record["info"]
                else:
                    return ''
        except exceptions.ServiceUnavailable as e:
            print("Connection failed:", e)
            return ''
# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    uri = "neo4j+s://<your-sandbox-host>:7687"
    user = "neo4j"
    password = "<your-password>"

    neo = Neo4jSandbox(uri, user, password)

    # Add or append info to Kire
    neo.add_info_to_person("Kire", "This is some text about Kire. ")

    # You can call it again, it will append
    neo.add_info_to_person("Kire", "More additional text. ")

    neo.close()

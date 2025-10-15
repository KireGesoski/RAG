from db_connect import Neo4jSandbox
class DataDB:
    def __init__(self, uri: str, user: str, password: str):
        self.db = Neo4jSandbox(uri, user, password)

    def create_user(self, user_id, name, surname, country, additional, options):
        """
        Wrapper for Neo4jSandbox.create_user_with_relations()
        """
        self.db.create_user_with_relations(
            user_id=user_id,
            name=name,
            surname=surname,
            country=country,
            additional=additional,
            options=options
        )
    def set_user_property(self, user_id, node_type, property_name, text_to_add):
        self.db.append_to_user_node_property(user_id, node_type, property_name, text_to_add)

    def get_user_info(self, user_id, node_type, property_name):
        return self.db.get_user_info(user_id, node_type, property_name)

    def close(self):
        """Close DB connection"""
        self.db.close()

if __name__ == "__main__":
    uri = "bolt://44.222.97.237"
    user = "neo4j"
    password = "wraps-injection-legs"
    importer = DataDB(uri, user, password)

    # importer.create_user(
    #     user_id="U1001",
    #     name="Kire",
    #     surname="Geshoski",
    #     country="Macedonia",
    #     additional="AI Engineer, enjoys cycling",
    #     options="Standard"
    # )

    #importer.set_user_property("U1001", "references", "FieldOfInterest", "Computers")
    info = importer.get_user_info("U1001", "references", "FieldOfInterest")
    print(info)
    importer.close()
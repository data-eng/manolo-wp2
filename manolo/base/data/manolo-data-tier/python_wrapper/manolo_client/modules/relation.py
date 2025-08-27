import json

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manolo_client.client import ManoloClient
class RelationMixin:
    def add_child(self: "ManoloClient" , dsn: int, Child: str, Parent: str):
        """Add a child node to a parent."""
        self.logger.debug(f"Adding child {Child} to parent {Parent}")
        params = {"Dsn": dsn, "Child": Child, "Parent": Parent}
        response = self.session.post(self._url("addChild"), params=params)
        return self._check_response(response)

    def add_edge(self: "ManoloClient" , dsn: int, Node1: str, Node2: str, IsDirected: int, Value: str):
        """Add an edge between two nodes."""
        self.logger.debug(f"Adding edge between {Node1} and {Node2}")
        params = {
            "Dsn": dsn,
            "Node1": Node1,
            "Node2": Node2,
            "IsDirected": IsDirected,
            "Value": Value
        }
        response = self.session.post(self._url("addEdge"), params=params)
        return self._check_response(response)

    def create_relation(self: "ManoloClient" , dsn: int, subject: str, predicate: str, object: str):
        """Create a relation between subject and object via predicate."""
        self.logger.debug(
            f"Creating relation between {subject} and {object} via {predicate}")
        params = {"Dsn": dsn, "Subject": subject,
                  "Predicate": predicate, "Object": object}
        response = self.session.post(
            self._url("createRelation"), params=params)
        return self._check_response(response)

    def delete_relation(self: "ManoloClient" , dsn: int, subject: str, predicate: str, object: str):
        """Delete a specific relation."""
        self.logger.debug(
            f"Deleting relation between {subject} and {object} via {predicate}")
        params = {"Dsn": dsn, "Subject": subject,
                  "Predicate": predicate, "Object": object}
        response = self.session.delete(
            self._url("deleteRelation"), params=params)
        return self._check_response(response)

    def get_adjacency_list(self: "ManoloClient" , node: str):
        """Get adjacency list of a node."""
        self.logger.debug(f"Getting adjacency list for node {node}")
        response = self.session.get(
            self._url("getAdjacencyList"), params={"Node": node})
        return json.loads(response.text.strip())

    def get_children(self: "ManoloClient" , parent: str):
        """Get children of a parent node."""
        self.logger.debug(f"Getting children for parent {parent}")
        response = self.session.get(
            self._url("getChildren"), params={"Parent": parent})
        return json.loads(response.text.strip())

    def get_objects_of(self: "ManoloClient" , subject: str, description: str):
        """Get objects for a subject based on a predicate description."""
        self.logger.debug(
            f"Getting objects for subject {subject} with predicate {description}")
        params = {"Subject": subject, "Description": description}
        response = self.session.get(self._url("getObjectsOf"), params=params)
        return json.loads(response.text.strip())

    def get_relations(self):
        """Retrieve all relations."""
        self.logger.debug("Getting all relations")
        response = self.session.get(self._url("getRelations"))
        return json.loads(response.text.strip())

    def get_subjects_of(self: "ManoloClient" , object: str, description: str):
        """Get subjects for a given object and predicate description."""
        self.logger.debug(
            f"Getting subjects for object {object} with predicate {description}")
        params = {"Object": object, "Description": description}
        response = self.session.get(self._url("getSubjectsOf"), params=params)
        return json.loads(response.text.strip())

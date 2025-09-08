from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import ManoloClient
class PredicateMixin:
    def create_predicate(self: "ManoloClient" , description: str):
        """Create a new predicate."""
        self.logger.debug(f"Creating predicate {description}")
        response = self.session.post(self._url("createPredicate"), params={
                                     "Description": description})
        return self._check_response(response)

    def delete_predicate(self: "ManoloClient" , description: str):
        """Delete a predicate."""
        self.logger.debug(f"Deleting predicate {description}")
        response = self.session.delete(self._url("deletePredicate"), params={
                                       "Description": description})
        return self._check_response(response)

    def get_objects_of_predicate(self: "ManoloClient" , description: str):
        """Get objects associated with a predicate."""
        self.logger.debug(f"Getting objects for predicate {description}")
        response = self.session.get(self._url("getObjectsOfPredicate"), params={
                                    "Description": description})
        return self._check_response(response)

    def get_predicates(self):
        """Get all predicates."""
        self.logger.debug("Getting all predicates")
        response = self.session.get(self._url("getPredicates"))
        return self._check_response(response)

    def get_subjects_of_predicate(self: "ManoloClient" , description: str):
        """Get subjects associated with a predicate."""
        self.logger.debug(f"Getting subjects for predicate {description}")
        response = self.session.get(self._url("getSubjectsOfPredicate"), params={
                                    "Description": description})
        return self._check_response(response)

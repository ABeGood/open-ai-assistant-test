"""
File containing implementations of Observer design pattern
"""

class Subscriber: 

    def update(publisher, topic: str):
        """
        Updates the state of the subscriber.

        Is called when `publisher` changes its `topic`.
        """
        raise Exception('Unimplement abstract funtion')

class Publisher:

    def __init__(self) -> None:
        self._subscribers: list[tuple[Subscriber, str]] = []

    def attach(self, subscriber: Subscriber, topic: str):
        """
        Attaches subscriber to the list listening for the topic.
        """
        self._subscribers.append((subscriber, topic))

    def detach(self, subscriber: Subscriber, topic: str | None) -> None:
        """
        Detaches subscriber from the list for the topic. If topic is not specified,
        then it detaches it from all of them
        """
        subs_to_keep = []

        for sub, top in self._subscribers:
            if sub is subscriber and topic is None or topic == top:
                pass
            else:
                subs_to_keep.append((sub, top))

        self._subscribers = subs_to_keep

    def notify(self, changed_topic: str):
        """
        Notifies the subscribers regarding change.
        """
        for sub, top in self._subscribers:
            if changed_topic == top:
                sub.update(self, top)
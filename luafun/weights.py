from collections import defaultdict

from luafun.utils.ring import RingBuffer


class WeigthStorage:
    """Store/retrieve the latest weight for a given model
    
    Notes
    -----
    This is a simple local implementation, remote storage could be used when
    training in a distributed system
    """

    def __init__(self, capacity):
        self.storage = defaultdict(lambda: RingBuffer(capacity, None))

    def __getitem__(self, model, index):
        """Retrieve model weights
        
        Parameters
        ----------
        model: str
            model hash id
            
        index: int    
            version offset (-1 = latest, 0 = oldest)
        """
        ring = self.storage[model]
        return ring[index]

    def __setitem__(self, model, weight):
        self.storage[model].append(weight)


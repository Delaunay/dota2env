from dataclasses import dataclass


@dataclass
class ProcessingStates:
    proto_rcv: float = -1
    proto_decode: float = -1
    proto_send: float = -1
    state_rcv: float = -1
    state_applied: float = -1
    state_replied: float = -1
    batch_generate: float = -1
    count: float = 1
    delay: float = 0
    acquire_time: float = 0

    def ready(self):
        return all((
            self.proto_rcv > 0,
            self.proto_decode > 0,
            self.proto_send > 0,
            self.state_rcv > 0,
            self.batch_generate > 0,
            self.state_applied > 0,
            self.state_replied > 0,
        ))

    def avg(self, name):
        if self.count < 1:
            return 0

        return getattr(self, name) / (self.count - 1)

    @property
    def total(self):
        return (
            # self.proto_rcv +
            self.proto_decode +
            self.proto_send +
            self.state_rcv +
            self.state_applied +
            self.batch_generate +
            self.state_replied)

    def add(self, other, state_replied):

        if other.ready():
            # skip the first observation
            if self.count > 0:
                # (other.proto_rcv - prev)
                self.delay = other.proto_rcv - state_replied
                self.proto_rcv += (other.proto_decode - other.proto_rcv)            # start decode - start rcv
                self.proto_decode += (other.proto_send - other.proto_decode)        # start sending - start decode
                self.proto_send += (other.state_rcv - other.proto_send)             # end send - start sending
                self.state_applied += (other.state_applied - other.state_rcv)       # end state applied - end send
                self.state_replied += (other.state_replied - other.state_applied)   # end replied - end applied

            self.count += other.count

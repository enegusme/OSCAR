import random

class ReplayBuffer:
    def __init__(self, size: int):
        self.buffer = []
        self.size = size
        self.position = 0

    def insert(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.size

    def sample(self, batch_size: int) -> tuple[list, list, list, list, list]:
        if batch_size < len(self.buffer):
            batch = random.sample(self.buffer, batch_size)
        else:
            batch = random.sample(self.buffer, len(self.buffer))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def insert_batch(self, states, actions, rewards, next_states, dones):
        batch_size = len(states)

        if len(self.buffer) < self.size:
            self.buffer.extend(zip(states, actions, rewards, next_states, dones))
        else:
            end_position = self.position + batch_size

            if end_position <= self.size:
                self.buffer[self.position:end_position] = zip(states, actions, rewards, next_states, dones)
            else:
                overflow = end_position - self.size
                self.buffer[self.position:] = zip(states[:batch_size - overflow], actions[:batch_size - overflow],
                                                  rewards[:batch_size - overflow], next_states[:batch_size - overflow],
                                                  dones[:batch_size - overflow])
                self.buffer[:overflow] = zip(states[batch_size - overflow:], actions[batch_size - overflow:],
                                             rewards[batch_size - overflow:], next_states[batch_size - overflow:],
                                             dones[batch_size - overflow:])

            self.position = (self.position + batch_size) % self.size


    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.__init__(self.size)

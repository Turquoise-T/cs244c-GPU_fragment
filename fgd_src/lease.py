# PAPER[§5] "Lease: specifies max_steps and max_duration for a scheduling round"
class Lease:
    def __init__(self, max_steps: int, max_duration: float):
        self._max_steps = max_steps
        self._max_duration = max_duration

    def __str__(self):
        return f'Lease(max_steps={self._max_steps}, max_duration={self._max_duration:.2f})'

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, steps):
        self._max_steps = steps

    @property
    def max_duration(self):
        return self._max_duration

    @max_duration.setter
    def max_duration(self, duration):
        self._max_duration = duration
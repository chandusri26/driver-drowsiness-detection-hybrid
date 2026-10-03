import winsound


class Alarm:

    def __init__(self, alarm_file="alarm.wav"):

        self.alarm_file = alarm_file
        self.is_playing = False


    def start(self):

        if not self.is_playing:

            winsound.PlaySound(
                self.alarm_file,
                winsound.SND_FILENAME |
                winsound.SND_ASYNC |
                winsound.SND_LOOP
            )

            self.is_playing = True


    def stop(self):

        if self.is_playing:

            winsound.PlaySound(
                None,
                winsound.SND_PURGE
            )

            self.is_playing = False
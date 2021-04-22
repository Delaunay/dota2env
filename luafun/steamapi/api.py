import time


class LimitExceeded(RuntimeError):
    pass


class ServerError(RuntimeError):
    pass


class WebAPI:
    """
"""
    URL = ''

    def __init__(self):
        # 100,000 API calls per day.
        # 1 request per second
        # 60 request per minute
        self.max_api_call_day = 100000
        self.start = None
        # make sure we respect the T&C of valve and do not get banned
        self.wait_time = 1
        self.limiter = True
        self.request_count = 0

    def limit_stats(self):
        return self.request_count / self.max_api_call_day

    def handle_errors(self, response):
        if response.status_code == 503:
            time.sleep(30)
            raise ServerError

        if response.status_code != 200:
            print(f'[API] Received `{response.reason}`')
            time.sleep(30)
            raise ServerError

    def limit(self):
        if self.limiter:
            # sleep a second to never go over the 1 request per second limit
            time.sleep(self.wait_time)
            self.request_count += 1

            if self.start is None:
                self.start = time.time()

            # reset the request count to 0 after a day
            if time.time() - self.start > 24 * 60 * 60:
                self.request_count = 0

            if self.request_count > self.max_api_call_day:
                raise LimitExceeded('Cannot make more requests today')

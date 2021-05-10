# A UTC class that allows the construction of datetime objects with the timezone of UTC.
# Tim Molteno 2013
#

from datetime import tzinfo, timedelta, datetime, date
import pytz

class UTC(tzinfo):
    """UTC"""
    ZERO = timedelta(0)
    HOUR = timedelta(hours=1)


    def utcoffset(self, dt):
        return self.ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return self.ZERO


def utc_datetime(year, month, day, hour=0, minute=0, second=0.0):
    s = int(second)
    us = int((second - int(second)) * 1000000)
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=s, microsecond=us, tzinfo=UTC())

def now():
    t = datetime.now(UTC())
    return t


def to_utc(dt):
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(pytz.utc)

def yday(d):
    # return day of year
    return d.toordinal() - date(d.year, 1, 1).toordinal() + 1

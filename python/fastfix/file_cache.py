import traceback
import logging
import datetime
import os
import urllib.request

from .ephemeris import Ephemerides
from .utc import to_utc, yday

class FileCache:
    def __init__(self, name):
        self.cache_root = f"./orbit_data/{name}"
        self.last_download_attempt = {}
        self.cache = {}
    
    def get_url(self, utc_date):
        doy = "%.3d" % yday(utc_date)
        yy = "%.2d" % (utc_date.year-2000)
        yyyy = utc_date.year
        #path = f"daily/{yyyy}/brdc/brdc{doy}0.{yy}n"
        #return f"ftp://cddis.gsfc.nasa.gov/gps/data/{path}"
        
        # ftp://igs.ensg.ign.fr/pub/igs/data/2001/012/brdc0120.01n.Z
        return f"ftp://igs.ensg.ign.fr/pub/igs/data/{yyyy}/{doy}/brdc{doy}0.{yy}n.Z"
    
    
    def get_local_filename(self, utc_date):
        return "{}/{}/{}.dat.Z".format(utc_date.year, utc_date.month,utc_date.day)

    def get_local_path(self, fname):
        return f"{self.cache_root}/{fname}"

    def create_object_from_file(self, local_path):
        # Override to create the object from the file
        pass

    def download_file(self, url, local_file):
        try:
            os.makedirs(os.path.dirname(local_file))
        except:
            pass
        try:
            if (url in self.last_download_attempt):
                logging.info(f"Download Attempt {self.last_download_attempt}")
                last_try = self.last_download_attempt[url]
                print("last_try {}".format(last_try))
                delta_seconds = (datetime.datetime.now() - last_try).total_seconds()
                if last_try and (delta_seconds < 3600):
                    raise RuntimeError(f"Error ({url} -> {local_file}: Already attempted ({last_try})")

            logging.info("starting download ({} -> {}".format(url, local_file))
            self.last_download_attempt[url] = datetime.datetime.now()
            dat = urllib.request.urlopen(url)
            with open(local_file, 'wb') as w:
                w.write(dat.read())
                w.close()
            logging.info("download complete")
        except Exception as error:
            tb = traceback.format_exc()
            logging.error(tb)
            self.last_download_attempt[url] = datetime.datetime.now()
            raise(error)
            

    def get_object(self, date):
        utc_date = to_utc(date)

        fname = self.get_local_filename(utc_date)
        if fname in self.cache:
            return self.cache[fname]

        try:
            local_path = self.get_local_path(fname)

            if (False == os.path.isfile(local_path)):
                self.download_file(self.get_url(utc_date), local_path)

            self.cache[fname] = self.create_object_from_file(local_path)
            return self.cache[fname]
        except Exception as error:
            # Something went horribly wrong. print(out the exception and use data from a day ago)
            tb = traceback.format_exc()
            logging.error(tb)
            logging.error("Something went wrong. Using old orbit information")
            return self.get_object(date -  datetime.timedelta(days=1))


class GPSFileCache(FileCache):
    ''' A proxy for requesting BRDC (Broadcast Ephemerides files) to avoid spamming external servers
    '''
    def __init__(self):
        FileCache.__init__(self, "brdc")

    def get_ephemerides(self, utc_date):
        eph = self.get_object(utc_date)
        return eph

    def create_object_from_file(self, local_path):
        return Ephemerides(local_path)

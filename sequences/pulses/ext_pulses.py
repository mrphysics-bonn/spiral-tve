
"""
Class to load external RF pulse from Siemens .pta files. These files can be created by using the export function
.pta files can be obtained by using the export function of the pulsetool
"""

import numpy as np

class Ext_pulse:

    def __find_char(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    @staticmethod
    def __isfloat(value):
        """
        Returns if string contains a valid float
        :param value: String
        :return: Float?
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def __init__(self, filename):
        """
        Load external RF pulse from Siemens .pta files. These files can be created by using the export function
        """

        # read pulse parameters
        self.hdr = {}
        self.rf_mag = []
        self.rf_phs = []

        # parameters for pulse calculations
        self.rf_raster = 1e-6
        self.rf = np.array([]) # on rf_raster, amp in [Hz], but not scaled for flip angle
        self.grad_amp = 0 # [Hz/m]

        with open(filename, 'r') as f:
            for line in f:
                if ':' in line:
                    key = line[:self.__find_char(line, ':')[0]].strip()
                    value = line[self.__find_char(line, '\t')[0]:].strip()
                    if self.__isfloat(value):
                        self.hdr[key] = float(value)
                    else:
                        self.hdr[key] = value
                elif ';' in line:
                    self.test = line
                    mag_ix = self.__find_char(line, '\t')[0]
                    phs_ix = self.__find_char(line, '\t')[1]
                    self.rf_mag.append(float(line[:mag_ix].strip()))
                    self.rf_phs.append(float(line[mag_ix:phs_ix].strip()))

    def calc_rf_grad(self, duration, slc_thickness=None, raster=1e-6):
        """
        Interpolate waveform to specified pulse duration and raster

        Calculate gradient amplitude [Hz/m] for slice selection, if slice thickness is given
        """

        self.rf_raster = raster

        rf = np.asarray(self.rf_mag) * np.exp(1j*np.asarray(self.rf_phs))

        n_samples_old = len(self.rf_mag)
        n_samples_new = int(duration/self.rf_raster)
        if n_samples_new > n_samples_old:
            div_sample = n_samples_old / n_samples_new
            oldgrid = np.linspace(1+div_sample,n_samples_new-div_sample, n_samples_old)
            newgrid = np.linspace(1,n_samples_new, n_samples_new)
        else:
            div_sample = n_samples_new / n_samples_old
            oldgrid = np.linspace(1,n_samples_old, n_samples_old)
            newgrid = np.linspace(1+div_sample,n_samples_old-div_sample, n_samples_new)

        self.rf = np.interp(newgrid, oldgrid, rf)

        if slc_thickness is not None:
            # REFGRAD is the gradient amplitude [mT/m] for a 10 mm slice thickness 
            # assuming a pulse duration of 5.12 msec.
            gamma = 42.57e6
            amp = self.hdr['REFGRAD'] * (10e-3*5.12e-3) / (slc_thickness*duration)
            self.grad_amp = amp * 1e-3 * gamma # to [Hz/m]

    def scale_rf(self, flip_angle):
        """
        Scale the RF pulse according to the desired flip angle

        flip_angle: flip_angle in [rad]

        returns RF in [Hz]
        """

        if not len(self.rf):
            raise ValueError("RF pulse not yet calculated, use calc_rf_grad first.")

        flip = np.sum(self.rf) *  self.rf_raster * 2 * np.pi

        return self.rf * flip_angle/flip

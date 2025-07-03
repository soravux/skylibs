import numpy as np
import datetime as dt
import xml.etree.ElementTree as ET


class EnvmapXMLParser:
    """
    Parser for the metadata file ( filename.meta.xml ).
    """
    def __init__(self, filename):
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()


    def _getFirstChildTag(self, tag):
        for elem in self.root:
            if elem.tag == tag:
                return elem.attrib


    def _getAttrib(self, node, attribute, default=None):
        if node:
            return node.get(attribute, default)
        return default


    def getFormat(self):
        """Returns the format of the environment map."""
        node = self._getFirstChildTag('data')
        return self._getAttrib(node, 'format', 'Unknown')


    def getDate(self):
        """Returns the date of the environment map in dict format."""
        return self._getFirstChildTag('date')


    def get_datetime(self):
        """Returns the date of the environment map in datetime format."""
        # Example: <date day="24" hour="16" minute="36" month="9" second="49.1429" utc="-4" year="2014"/>
        date = self.root.find('date')
        year = date.get('year')
        month = date.get('month').zfill(2)
        day = date.get('day').zfill(2)
        hour = date.get('hour').zfill(2)
        minute = date.get('minute').zfill(2)
        second = str(int(float(date.get('second')))).zfill(2)
        utc_offset = int(date.get('utc'))
        if utc_offset > 0:
            utc_offset = f'+{str(utc_offset).zfill(2)}'
        else:
            utc_offset = f'-{str(np.abs(utc_offset)).zfill(2)}'
        return dt.datetime.fromisoformat(f"{year}-{month}-{day} {hour}:{minute}:{second}{utc_offset}:00")


    def get_calibration(self):
        """Returns the OCamCalib calibration metadata."""

        calibration = {}
        # Calibration Model
        node = self.root.find('calibrationModel')
        # Affine 2D = [c,d;
        #              e,1];
        c = float(node.get('c'))
        d = float(node.get('d'))
        e = float(node.get('e'))
        affine_2x2 = np.array([[c,d],[e, 1]])
        calibration['c'] = c
        calibration['d'] = d
        calibration['e'] = e
        calibration['affine_2x2'] = affine_2x2

        # shape = [height, width]
        height = int(node.get('height'))
        width = int(node.get('width'))
        calibration['height'] = height
        calibration['width'] = width
        calibration['shape'] = (height, width)

        # center = [xc, yc]
        xc = float(node.get('xc'))
        yc = float(node.get('yc'))
        calibration['xc'] = xc
        calibration['yc'] = yc
        calibration['center'] = (xc, yc)

        # Affine 3D = [c,d,xc;
        #              e,1,yc;
        #              0,0,1];
        affine_3x3 = np.array([[c,d,xc],[e, 1, yc],[0,0,1]])
        calibration['affine_3x3'] = affine_3x3

        # ss = [a_0, a_1, ..., a_n]
        ss = [ float(s.get('s')) for s in node.findall('ss') ]
        calibration['ss'] = np.array(ss)
        polydomain = xc if xc > yc else yc
        polydomain = [-polydomain,polydomain]
        calibration['F'] = \
            np.polynomial.polynomial.Polynomial(
                ss,
                domain=polydomain,
                window=polydomain
            )

        return calibration


    def getExposure(self):
        """Returns the exposure of the environment map in EV."""
        node = self._getFirstChildTag('exposure')
        return float(self._getAttrib(node, 'EV'))

import array


def reflect(num, width):
    """Reverts bit order of the given number
    Args:
        num (int): Number that should be reflected
        width (int): Size of the number in bits
    """
    reflected = 0

    for i in range(width):
        if (num >> i) & 1 != 0:
            reflected |= 1 << (width - 1 - i)

    return reflected


def make_table(width):
    """Create static sized CRC lookup table and initialize it with ``0``.
    For 8, 16, 32, and 64 bit width :class:`array.array` instances are used. For
    all other widths it falls back to a generic Python :class:`list`.
    Args:
        width (int): Size of elements in bits
    """
    initializer = (0 for _ in range(256))

    if width == 8:
        return array.array('B', initializer)
    elif width == 16:
        return array.array('H', initializer)
    elif width == 32:
        return array.array('L', initializer)
    elif width == 64:
        return array.array('Q', initializer)
    else:
        # Fallback to a generic list
        return list(initializer)


class CRC(object):
    """Generic CRC model implemented with lookup tables.
    The model parameter can are the constructor parameters.
    Args:
        width (int): Number of bits of the polynomial
        polynomial (int): CRC polynomial
        initial_value (int): Initial value of the checksum
        final_xor_value (int): Value that will be XOR-ed with final checksum
        input_reflected (bool): True, if each input byte should be reflected
        result_reflected (bool): True, if the result should be reflected before
            the final XOR is applied
    Usage:
        .. code:: python
            from crc import CRC
            # Definition  of CRC-32 Ethernet. The crc module defines many common
            # models already.
            crc = CRC(32, 0x04c11db7, 0xffffffff, 0xffffffff, True, True)
            # You can call the model to calculate the CRC checksum of byte
            # string
            assert crc(b"Hello world!") == 0x1b851995
    """

    def __init__(self, width, polynomial, initial_value, final_xor_value, input_reflected, result_reflected):
        self.width = width
        self.polynomial = polynomial
        self.initial_value = initial_value
        self.final_xor_value = final_xor_value
        self.input_reflected = input_reflected
        self.result_reflected = result_reflected

        # Initialize casting mask to keep the correct width for dynamic Python
        # integers
        self.cast_mask = int('1' * self.width, base=2)

        # Mask that can be applied to get the Most Significant Bit (MSB) if the
        # number with given width
        self.msb_mask = 0x01 << (self.width - 1)

        # The lookup tables get initialized lazzily. This ensures that only
        # tables are calculated that are actually needed.
        self.table = None
        self.reflected_table = None

    def __call__(self, value):
        """Compute the CRC checksum with respect to the model parameters by using
        a looup table algorithm.
        Args:
            value (bytes): Input bytes that should be checked
        Returns:
            int - CRC checksum
        """
        # Use the reflection optimization if applicable
        if self.input_reflected and self.result_reflected:
            return self.fast_reflected(value)

        # Lazy initialization of the lookup table
        if self.table is None:
            self.table = self.calculate_crc_table()

        crc = self.initial_value

        for cur_byte in value:
            if self.input_reflected:
                cur_byte = reflect(cur_byte, 8)

            # Update the MSB of the CRC value with the next input byte
            crc = (crc ^ (cur_byte << (self.width - 8))) & self.cast_mask

            # This MSB byte value is the index into the lookup table
            index = (crc >> (self.width - 8)) & 0xff

            # Shift out the index
            crc = (crc << 8) & self.cast_mask

            # XOR-ing crc from the lookup table using the calculated index
            crc = crc ^ self.table[index]

        if self.result_reflected:
            crc = reflect(crc, self.width)

        # Final XBOR
        return crc ^ self.final_xor_value

    def fast_reflected(self, value):
        """If the input data and the result checksum are both reflected in the
        current model, an optimized algorithm can be used that reflects the
        looup table rather then the input data. This saves the reflection
        operation of the input data.
        """
        if not self.input_reflected or not self.result_reflected:
            raise ValueError("Input and result must be reflected")

        # Lazy initialization of the lookup table
        if self.reflected_table is None:
            self.reflected_table = self.calculate_crc_table_reflected()

        crc = self.initial_value

        for cur_byte in value:
            # The LSB of the XOR-red remainder and the next byte is the index
            # into the lookup table
            index = (crc & 0xff) ^ cur_byte

            # Shift out the index
            crc = (crc >> 8) & self.cast_mask

            # XOR-ing remainder from the loopup table
            crc = crc ^ self.reflected_table[index]

        # Final XBOR
        return crc ^ self.final_xor_value

    def calculate_crc_table(self):
        table = make_table(self.width)

        for divident in range(256):
            cur_byte = (divident << (self.width - 8)) & self.cast_mask

            for bit in range(8):
                if (cur_byte & self.msb_mask) != 0:
                    cur_byte <<= 1
                    cur_byte ^= self.polynomial
                else:
                    cur_byte <<= 1

            table[divident] = cur_byte & self.cast_mask

        return table

    def calculate_crc_table_reflected(self):
        table = make_table(self.width)

        for divident in range(256):
            reflected_divident = reflect(divident, 8)
            cur_byte = (reflected_divident << (self.width - 8)) & self.cast_mask

            for bit in range(8):
                if (cur_byte & self.msb_mask) != 0:
                    cur_byte <<= 1
                    cur_byte ^= self.polynomial
                else:
                    cur_byte <<= 1

            cur_byte = reflect(cur_byte, self.width)

            table[divident] = (cur_byte & self.cast_mask)

        return table


# Known CRC algorihtms
crc8 = CRC(8, 0x07, 0x00, 0x00, False, False)
crc8_sae_j1850 = CRC(8, 0x1d, 0xff, 0xff, False, False)
crc8_sae_j1850_zero = CRC(8, 0x1d, 0x00, 0x00, False, False)
crc8_8h2f = CRC(8, 0x2f, 0xff, 0xff, False, False)
crc8_cdma2000 = CRC(8, 0x9b, 0xff, 0x00, False, False)
crc8_darc = CRC(8, 0x39, 0x00, 0x00, True, True)
crc8_dvb_s2 = CRC(8, 0xd5, 0x00, 0x00, False, False)
crc8_ebu = CRC(8, 0x1d, 0xff, 0x00, True, True)
crc8_icode = CRC(8, 0x1d, 0xfd, 0x00, False, False)
crc8_itu = CRC(8, 0x07, 0x00, 0x55, False, False)
crc8_maxim = CRC(8, 0x31, 0x00, 0x00, True, True)
crc8_rohc = CRC(8, 0x07, 0xff, 0x00, True, True)
crc8_wcdma = CRC(8, 0x9b, 0x00, 0x00, True, True)

crc16_ccit_zero = CRC(16, 0x1021, 0x0000, 0x0000, False, False)
crc16_arc = CRC(16, 0x8005, 0x0000, 0x0000, True, True)
crc16_aug_ccitt = CRC(16, 0x1021, 0x1d0f, 0x0000, False, False)
crc16_buypass = CRC(16, 0x8005, 0x0000, 0x0000, False, False)
crc16_ccitt_false = CRC(16, 0x1021, 0xffff, 0x0000, False, False)
crc16_cdma2000 = CRC(16, 0xc867, 0xffff, 0x0000, False, False)
crc16_dds_110 = CRC(16, 0x8005, 0x800d, 0x0000, False, False)
crc16_dect_r = CRC(16, 0x0589, 0x0000, 0x0001, False, False)
crc16_dect_x = CRC(16, 0x0589, 0x0000, 0x0000, False, False)
crc16_dnp = CRC(16, 0x3d65, 0x0000, 0xffff, True, True)
crc16_en_13757 = CRC(16, 0x3d65, 0x0000, 0xffff, False, False)
crc16_genibus = CRC(16, 0x1021, 0xffff, 0xffff, False, False)
crc16_maxim = CRC(16, 0x8005, 0x0000, 0xffff, True, True)
crc16_mcrf4xx = CRC(16, 0x1021, 0xffff, 0x0000, True, True)
crc16_riello = CRC(16, 0x1021, 0xb2aa, 0x0000, True, True)
crc16_t10_dif = CRC(16, 0x8bb7, 0x0000, 0x0000, False, False)
crc16_teledisk = CRC(16, 0xa097, 0x0000, 0x0000, False, False)
crc16_tms37157 = CRC(16, 0x1021, 0x89ec, 0x0000, True, True)
crc16_usb = CRC(16, 0x8005, 0xffff, 0xffff, True, True)
crc16_a = CRC(16, 0x1021, 0xc6c6, 0x0000, True, True)
crc16_kermit = CRC(16, 0x1021, 0x0000, 0x0000, True, True)
crc16_modbus = CRC(16, 0x8005, 0xffff, 0x0000, True, True)
crc16_x25 = CRC(16, 0x1021, 0xffff, 0xffff, True, True)
crc16_xmodem = CRC(16, 0x1021, 0x0000, 0x0000, False, False)

crc32 = CRC(32, 0x04c11db7, 0xffffffff, 0xffffffff, True, True)
crc32_bzip2 = CRC(32, 0x04c11db7, 0xffffffff, 0xffffffff, False, False)
crc32_c = CRC(32, 0x1edc6f41, 0xffffffff, 0xffffffff, True, True)
crc32_d = CRC(32, 0xa833982b, 0xffffffff, 0xffffffff, True, True)
crc32_mpeg2 = CRC(32, 0x04c11db7, 0xffffffff, 0x00000000, False, False)
crc32_posix = CRC(32, 0x04c11db7, 0x00000000, 0xffffffff, False, False)
crc32_q = CRC(32, 0x814141ab, 0x00000000, 0x00000000, False, False)
crc32_jamcrc = CRC(32, 0x04c11db7, 0xffffffff, 0x00000000, True, True)
crc32_xfer = CRC(32, 0x000000af, 0x00000000, 0x00000000, False, False)


# ----------
# Unit tests
# ----------

def test_crc16_cc():
    assert crc16_ccit_zero(b'Hello world!') == 0x39db


def test_crc16_x25():
    assert crc16_x25(b'Hello world!') == 0x8edb


def test_crc32():
    assert crc32(b'Hello world!') == 0x1b851995

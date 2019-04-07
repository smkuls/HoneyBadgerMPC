from Crypto.Cipher import AES
from Crypto import Random
from hashlib import sha256


class SymmetricCrypto(object):
    """
    Uses AES with a 32-byte key.
    Semantic security (iv is randomized).
    Copied from honeybadgerbft.
    """

    BS = 16

    @staticmethod
    def pad(s):
        padding = (SymmetricCrypto.BS - len(s) % SymmetricCrypto.BS) * bytes([
            SymmetricCrypto.BS - len(s) % SymmetricCrypto.BS])
        return s + padding

    @staticmethod
    def unpad(s):
        return s[:-ord(s[len(s)-1:])]

    @staticmethod
    def encrypt(key, plaintext):
        """ """
        key = sha256(key).digest()  # hash the key
        assert len(key) == 32
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = (
            iv + cipher.encrypt(SymmetricCrypto.pad(serialize_tuple(plaintext))))
        return ciphertext

    @staticmethod
    def decrypt(key, ciphertext):
        """ """
        key = sha256(key).digest()  # hash the key
        assert len(key) == 32
        iv = ciphertext[:16]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = deserialize_tuple(
            SymmetricCrypto.unpad(cipher.decrypt(ciphertext[16:])))
        return plaintext


def serialize_tuple(value):
    assert type(value) is tuple
    assert len(value) == 2
    assert type(value[0]) is int
    assert type(value[1]) is int
    return (str(value[0]) + "," + str(value[1])).encode()


def deserialize_tuple(value):
    s = value.decode()
    values = s.split(",")
    return (int(values[0]), int(values[1]))

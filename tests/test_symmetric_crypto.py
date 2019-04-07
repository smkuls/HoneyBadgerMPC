import uuid
from honeybadgermpc.symmetric_crypto import SymmetricCrypto


def test_encrypt_decrypt(galois_field):
    key = uuid.uuid4().hex.encode('utf-8')
    plaintext = (galois_field.random().value, galois_field.random().value)
    ciphertext = SymmetricCrypto.encrypt(key, plaintext)
    plaintext_ = SymmetricCrypto.decrypt(key, ciphertext)
    assert plaintext_ == plaintext

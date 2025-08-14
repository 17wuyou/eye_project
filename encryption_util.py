# encryption_util.py
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
import base64
from typing import Union
import logging
import configs
logger = logging.getLogger(__name__)

AES_KEY_STRING = configs.AES_KEY_STRING

try:
    AES_KEY_BYTES_FROM_STRING = base64.b64decode(AES_KEY_STRING)
    
    # 将解码后的密钥字节转换为十六进制字符串以便比较
    hex_string = AES_KEY_BYTES_FROM_STRING.hex()
    logger.info(f"PYTHON Decoded AES_KEY Hex: {hex_string}") # <--- 关键日志

    if len(AES_KEY_BYTES_FROM_STRING) != 32:
        logger.critical(f"Decoded AES_KEY is not 32 bytes long! Length: {len(AES_KEY_BYTES_FROM_STRING)}. AES_KEY_STRING was: '{AES_KEY_STRING}'.")
        raise ValueError("Decoded AES_KEY is not 32 bytes long!")
    
    AES_KEY = AES_KEY_BYTES_FROM_STRING # 直接使用解码后的字节
    logger.info("AES_KEY successfully decoded from string and is 32 bytes long.")

except Exception as e:
    logger.critical(f"Failed to decode AES_KEY_STRING: '{AES_KEY_STRING}'. Error: {e}", exc_info=True)
    logger.critical("Application will likely fail all encryption/decryption. Please provide a valid Base64 encoded 32-byte key.")
    AES_KEY = os.urandom(32)
    logger.warning("FALLING BACK TO A TEMPORARY RANDOM KEY - THIS WILL NOT MATCH THE CLIENT KEY!")


GCM_IV_LENGTH = 12
GCM_TAG_LENGTH = 16 

def encrypt(plaintext_bytes: bytes) -> Union[bytes, None]:
    if AES_KEY == os.urandom(32): # A simple (though not foolproof) check if we're using the fallback
        logger.error("encrypt: Attempting to encrypt with a DUMMY/FALLBACK key. This encryption WILL NOT MATCH the other side if they have the correct key.")

    try:
        aesgcm = AESGCM(AES_KEY)
        iv = os.urandom(GCM_IV_LENGTH)
        ciphertext = aesgcm.encrypt(iv, plaintext_bytes, None)
        return iv + ciphertext
    except Exception as e:
        logger.error(f"Encryption failed for {len(plaintext_bytes)} bytes.", exc_info=True)
        return None

def decrypt(ciphertext_with_iv: bytes) -> Union[bytes, None]:
    if AES_KEY == os.urandom(32):
         logger.error("decrypt: Attempting to decrypt with a DUMMY/FALLBACK key. This WILL LIKELY FAIL.")
    try:
        if len(ciphertext_with_iv) < GCM_IV_LENGTH:
            logger.error("Decryption failed: Ciphertext too short to contain IV.")
            return None
        iv = ciphertext_with_iv[:GCM_IV_LENGTH]
        ciphertext = ciphertext_with_iv[GCM_IV_LENGTH:]
        aesgcm = AESGCM(AES_KEY)
        return aesgcm.decrypt(iv, ciphertext, None)
    except InvalidTag:
        logger.error("Decryption failed: Invalid authentication tag (likely wrong key or corrupted data).")
        return None
    except Exception as e:
        logger.error(f"Decryption failed with an unexpected error.", exc_info=True)
        return None

def encrypt_to_string(plaintext_str: str) -> Union[str, None]:
    encrypted_bytes = encrypt(plaintext_str.encode('utf-8'))
    if encrypted_bytes:
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    else:
        # logger.warning(f"encrypt_to_string: 'encrypt' function returned None for plaintext (partial): {plaintext_str[:70]}...")
        pass # encrypt() already logs
    return None

def decrypt_from_string(base64_ciphertext_with_iv: str) -> Union[str, None]:
    try:
        ciphertext_with_iv = base64.b64decode(base64_ciphertext_with_iv)
        decrypted_bytes = decrypt(ciphertext_with_iv)
        if decrypted_bytes:
            return decrypted_bytes.decode('utf-8')
        else:
            # logger.warning(f"decrypt_from_string: 'decrypt' function returned None for base64_ciphertext (partial): {base64_ciphertext_with_iv[:70]}...")
            pass # decrypt() already logs
    except Exception as e:
        logger.error(f"Error in decrypt_from_string (decoding base64 or general error).", exc_info=True)
    return None
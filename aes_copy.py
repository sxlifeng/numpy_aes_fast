# coding: utf-8
import time
from enum import Enum

import numpy as np
from sbox import InvSbox, Rcon, SBOX, RconWord
from sbox import M1, M2, M3, M9, M11, M13, M14


class AesType(Enum):
    AES128 = (128, 4, 10)
    AES192 = (192, 6, 12)
    AES256 = (256, 8, 14)

    def __init__(self, key_bits, key_words, rounds):
        self.key_bits = key_bits
        self.key_words = key_words
        self.nk = key_words
        self.rounds = rounds


class AesMidType(Enum):
    SBOX_IN = 1
    SBOX_OUT = 2
    ROUND_IN = 3
    ROUND_OUT = 4
    SHIFT_ROWS = 5
    MIX_COLUMNS = 6
    XOR_SBOX_IN_OUT = 7
    XOR_ROUND_IN_OUT = 8


def hex2states(hex_str):
    d = int.to_bytes(int(hex_str, 16), length=len(hex_str) // 2, byteorder='big')
    s0 = np.frombuffer(d, 'u1')
    rest = (16 - (len(s0) % 16)) % 16
    s1 = np.r_[s0, np.zeros(rest)]
    return s1.astype('u1')


def subBytes(state):
    return SBOX[state]


def invSubBytes(state):
    return InvSbox[state]


def rShiftRows(state):
    res = np.zeros(state.shape, dtype=np.uint8)
    for i in range(4):
        res[:, :, i] = np.roll(state[:, :, i], i, axis=1)
    return res


def lShiftRows(state):
    res = np.zeros(state.shape, dtype=np.uint8)
    for i in range(4):
        res[:, :, i] = np.roll(state[:, :, i], -i, axis=1)
    return res


def mixColumn(state):
    s0 = state.reshape((-1, 4))
    d = np.zeros(s0.shape, dtype='u1')
    d[:, 0] = M2[s0[:, 0]] ^ M3[s0[:, 1]] ^ M1[s0[:, 2]] ^ M1[s0[:, 3]]
    d[:, 1] = M1[s0[:, 0]] ^ M2[s0[:, 1]] ^ M3[s0[:, 2]] ^ M1[s0[:, 3]]
    d[:, 2] = M1[s0[:, 0]] ^ M1[s0[:, 1]] ^ M2[s0[:, 2]] ^ M3[s0[:, 3]]
    d[:, 3] = M3[s0[:, 0]] ^ M1[s0[:, 1]] ^ M1[s0[:, 2]] ^ M2[s0[:, 3]]
    s1 = d.reshape([-1, 4, 4])
    return s1


def invMixColumn(state):
    s0 = state.reshape((-1, 4))
    d = np.zeros(s0.shape, dtype='u1')
    d[:, 0] = M14[s0[:, 0]] ^ M11[s0[:, 1]] ^ M13[s0[:, 2]] ^ M9[s0[:, 3]]
    d[:, 1] = M9[s0[:, 0]] ^ M14[s0[:, 1]] ^ M11[s0[:, 2]] ^ M13[s0[:, 3]]
    d[:, 2] = M13[s0[:, 0]] ^ M9[s0[:, 1]] ^ M14[s0[:, 2]] ^ M11[s0[:, 3]]
    d[:, 3] = M11[s0[:, 0]] ^ M13[s0[:, 1]] ^ M9[s0[:, 2]] ^ M14[s0[:, 3]]
    s1 = d.reshape([-1, 4, 4])
    return s1


def keySchedule(key, aes_type: AesType = AesType.AES128):
    # if len(key) != 16:
    #     raise ValueError("Only key with 16 byte length supported!")

    r0 = key.reshape((aes_type.key_words, 4)).tolist()
    for i in range(aes_type.key_words, 4 * (aes_type.rounds + 1)):
        r0.append([])

        if i % aes_type.key_words == 0:
            byte = r0[i - aes_type.nk][0] ^ SBOX[r0[i - 1][1]] ^ Rcon[i // aes_type.nk]
            r0[i].append(byte)

            for j in range(1, 4):
                byte = r0[i - aes_type.nk][j] ^ SBOX[r0[i - 1][(j + 1) % 4]]
                r0[i].append(byte)
        elif aes_type.key_words > 6 and i % aes_type.key_words == 4:
            for j in range(0, 4):
                byte = SBOX[r0[i - 1][j]] ^ r0[i - aes_type.nk][j]
                r0[i].append(byte)
        else:
            for j in range(4):
                byte = r0[i - aes_type.nk][j] ^ r0[i - 1][j]
                r0[i].append(byte)
    return np.array(r0).reshape((-1, 4, 4))


def keySchedule1(key, aes_type: AesType = AesType.AES128):
    # if len(key) != 16:
    #     raise ValueError("Only key with 16 byte length supported!")

    r0 = key.reshape([-1, aes_type.nk, 4])
    round_key = np.zeros([r0.shape[0], 4 * (aes_type.rounds + 1), 4], dtype=np.uint8)
    round_key[:, :aes_type.nk, :] = r0
    for i in range(aes_type.nk, 4 * (aes_type.rounds + 1)):
        # round_key[:, i, x]
        if i % aes_type.nk == 0:
            temp = SBOX[np.roll(round_key[:, i - 1, :], -1, axis=1)] ^ RconWord[i // aes_type.nk, :]
            round_key[:, i, :] = temp ^ round_key[:, i - aes_type.nk, :]
        elif aes_type.key_words > 6 and i % aes_type.key_words == 4:
            round_key[:, i, :] = np.bitwise_xor(SBOX[round_key[:, i - 1, :]], round_key[:, i - aes_type.nk, :])
        else:
            round_key[:, i, :] = np.bitwise_xor(round_key[:, i - 1, :], round_key[:, i - aes_type.nk, :])

    return round_key.reshape([-1, aes_type.rounds + 1, 4, 4])


def addRoundKey(state, key):
    return np.bitwise_xor(state, key)


def aes_encrypt(data, key, aes_type: AesType = AesType.AES128):
    roundKey = keySchedule1(key, aes_type)
    state = data.reshape([-1, 4, 4])
    s0 = addRoundKey(state, roundKey[:, 0])
    for i in range(1, aes_type.rounds + 1):
        s1 = subBytes(s0)
        s2 = lShiftRows(s1)
        s3 = mixColumn(s2) if i != aes_type.rounds else s2
        s0 = addRoundKey(s3, roundKey[:, i])
    s4 = s0.ravel().astype(np.uint8)
    return s4


def aes_encrypt_with_mid_value(data, key, mid_type: list or AesMidType, target_round: list or int):
    mid_value = {}
    if not isinstance(mid_type, list):
        mid_type = [mid_type]
    for mid in mid_type:
        mid_value[mid] = np.zeros([0, data.size], dtype=np.uint8)
    if not isinstance(target_round, list):
        target_round = [target_round]

    roundKey = keySchedule(key)
    state = data.reshape([-1, 4, 4])
    s0 = addRoundKey(state, roundKey[0])
    for i in range(1, 11):
        s1 = subBytes(s0)
        s2 = lShiftRows(s1)
        s3 = mixColumn(s2) if i != 10 else s2
        if i in target_round:
            if AesMidType.SBOX_IN in mid_type:
                mid_value[AesMidType.SBOX_IN] = np.concatenate((mid_value[AesMidType.SBOX_IN], s0.reshape([1, -1])))
            if AesMidType.SBOX_OUT in mid_type:
                mid_value[AesMidType.SBOX_OUT] = np.concatenate((mid_value[AesMidType.SBOX_OUT], s1.reshape([1, -1])))
            if AesMidType.SHIFT_ROWS in mid_type:
                mid_value[AesMidType.SHIFT_ROWS] = np.concatenate(
                    (mid_value[AesMidType.SHIFT_ROWS], s2.reshape([1, -1])))
            if AesMidType.MIX_COLUMNS in mid_type:
                mid_value[AesMidType.MIX_COLUMNS] = np.concatenate(
                    (mid_value[AesMidType.MIX_COLUMNS], s3.reshape([1, -1])))
            if AesMidType.XOR_SBOX_IN_OUT in mid_type:
                mid_value[AesMidType.XOR_SBOX_IN_OUT] = np.concatenate((mid_value[AesMidType.XOR_SBOX_IN_OUT],
                                                                        np.bitwise_xor(s0, s1).reshape([1, -1])))
        if i >= max(target_round):
            break
        s0 = addRoundKey(s3, roundKey[i])
    return mid_value


def aes_decrypt(cipher, key):
    roundKey = keySchedule(key)
    s0 = cipher.reshape([-1, 4, 4])
    for i in reversed(range(1, 11)):
        s3 = addRoundKey(s0, roundKey[i])
        s2 = invMixColumn(s3) if i != 10 else s3
        s1 = rShiftRows(s2)
        s0 = invSubBytes(s1)

    state = addRoundKey(s0, roundKey[0])
    plain = state.astype(np.uint8).ravel()

    return plain


def test_hex_data():
    key = "2b7e151628aed2a6abf7158809cf4f3c3243f6a8885a308d313198a2e0370734"
    key = np.frombuffer(int.to_bytes(int(key, 16), len(key) // 2, byteorder='big'), dtype=np.uint8)
    # keySchedule1(key)

    plain = "3243f6a8885a308d313198a2e03707343243f6a8885a308d313198a2e0370734"
    aes_type = AesType.AES128



    # key = "8E73B0F7 DA0E6452 C810F32B 809079E5 62F8EAD2 522C6B7B".replace(' ', '')
    # key = np.frombuffer(int.to_bytes(int(key, 16), len(key) // 2, byteorder='big'), dtype=np.uint8)
    # plain = "6BC1BEE2 2E409F96 E93D7E11 7393172A AE2D8A57 1E03AC9C 9EB76FAC 45AF8E51 30C81C46 A35CE411 E5FBC119 1A0A52EF F69F2445 DF4F9B17 AD2B417B E66C3710".replace(' ', '')
    # ref_cipher = "BD334F1D 6E45F25F F712A214 571FA5CC 97410484 6D0AD3AD 7734ECB3 ECEE4EEF EF7AFD22 70E2E60A DCE0BA2F ACE6444E 9A4B41BA 738D6C72 FB166916 03C18E0E".replace(' ', '')
    # aes_type = AesType.AES192
    #
    # key = "603DEB10 15CA71BE 2B73AEF0 857D7781 1F352C07 3B6108D7 2D9810A3 0914DFF4".replace(' ', '')
    # key = np.frombuffer(int.to_bytes(int(key, 16), len(key) // 2, byteorder='big'), dtype=np.uint8)
    # plain = "6BC1BEE2 2E409F96 E93D7E11 7393172A AE2D8A57 1E03AC9C 9EB76FAC 45AF8E51 30C81C46 A35CE411 E5FBC119 1A0A52EF F69F2445 DF4F9B17 AD2B417B E66C3710".replace(
    #     ' ', '')
    # ref_cipher = " F3EED1BD B5D2A03C 064B5A7E 3DB181F8 591CCB10 D410ED26 DC5BA74A 31362870 B6ED21B9 9CA6F4F9 F153E7B1 BEAFED1D 23304B7A 39F9F3FF 067D8D8F 9E24ECC7".replace(
    #     ' ', '')
    # aes_type = AesType.AES256

    # key = "000102030405060708090A0B0C0D0E0F"
    # plain = "d042c8fcd0351e1a1e9ad5c0c0c55afc3243f6a8885a308d313198a2e0370734"
    plain = hex2states(plain)
    # plain = np.random.randint(1, 255, [2, 16], dtype=np.uint8)
    start = time.time()
    # plain = raw_plain.reshape([1, -1])
    c = aes_encrypt(plain, key, aes_type)
    # mid = aes_encrypt_with_mid_value(plain, key, [AesMidType.SBOX_IN, AesMidType.SHIFT_ROWS, AesMidType.MIX_COLUMNS,
    #                                               AesMidType.SBOX_OUT, AesMidType.XOR_SBOX_IN_OUT], list(range(1, 11)))
    # mid[AesMidType.SBOX_IN].reshape([10, -1, 16]).transpose([1, 0, 2]).reshape([2, -1])
    print(time.time() - start)
    start = time.time()
    # p = aes_decrypt(c, key)
    # print(time.time() - start)
    print(bytes(c).hex())


test_hex_data()
# import matplotlib.pyplot as plt
#
# plain_file = r"D:\资料与文档\SideChannel\001_plain.txt"
# cipher_fle = r"D:\资料与文档\SideChannel\001_cipher.txt"
# sample_file = r"D:\资料与文档\SideChannel\001_trace_int.txt"
#
# plain = np.loadtxt(plain_file, dtype=np.uint8)
# cipher = np.loadtxt(cipher_fle, dtype=np.uint8)
# sample = np.loadtxt(sample_file, dtype=np.int32, max_rows=10000)
# key = "000102030405060708090A0B0C0D0E0F"
# # c = aes_encrypt(plain, key).reshape([-1, 16])
# # print(np.equal(cipher, c).all())
# # p = aes_decrypt(cipher, key).reshape([-1, 16])
# # print(np.equal(plain, p).all())
# start = time.time()
# mid = aes_encrypt_with_mid_value(plain[:10000], key, [AesMidType.XOR_SBOX_IN_OUT], [1, 3, 10])
# for mid_type, mid_value in mid.items():
#     # print(mid_type)
#     # print(mid_value.shape)
#     mid_value = mid_value.reshape([3, -1, 16]).transpose([1, 0, 2]).reshape([10000, -1])
#     # print(sample.shape, mid_value.shape)
#     corr = np.corrcoef(sample, mid_value, rowvar=False)[-32:, :3125]
#     # plt.plot(corr[0:16].transpose())
#     # plt.show()
#     # corr = Corr()
#     # corr.add_block(sample[1000:], mid_value[1000:])
#     # res = corr.calculate_corr()
#     # print(corr.shape)
# print(time.time() - start)
# print(plain.shape)

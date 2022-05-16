#
# class Solution:
#     def longestPalindrome(self, s: str) -> str:
#
#
#         length = len(s)
#
#         dp = [[False] * length for _ in range(length)]
#
#         for _ in range(length):
#             dp[_][_] = True
#         ll = -1
#         start = -1
#         for l in range(1, length):
#
#             i = 0
#             while i + l < length:
#
#                 if l== 1:
#                     dp[i][i + l] = (s[i] == s[i + l])
#                 else:
#                     dp[i][i + l] = dp[i + 1][i + l - 1] & (s[i] == [s + l])
#
#                 if dp[i][i + l] and l > ll:
#                     start = i
#                     ll = l
#                 i += 1
#         return s[start:start + ll + 1] if ll > 0 else ""
#
# print(Solution().longestPalindrome("babad")) 

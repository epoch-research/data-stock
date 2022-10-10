# Models of data stocks

This repository contains some probabilistic models in Squiggle, together with some JS code to run the models, especially computing the aggregated models and the intersection years, which are quite heavy.

Below is a summary of the models with links to the Squiggle playgrond. To view the model demos in the playground, open the link, click "View settings", scroll to "Function Display Settings" and set the min and max X value to 2022 and 2080, respectively. Some of them might take a few seconds to load.

## Language data

- Recorded human speech: uses the total words spoken by all humans as an upper bound on language data production. ([Squiggle demo](https://www.squiggle-language.com/playground/#code=eNqtWP1v20YS%2FVcWORSR7LW137sU0AL3VSBAe3e45JAf4sCgpZXFs0Sq%2FIitJr6%2F%2Fd4sSUmOk2sPiNtIw%2BXuzJs3b3ZJfXzRrKv71912m9f7F%2FO27iJPQ39dFm1VjyNFWbRFvnn9S1fc3m7i67YuytsX8xezGfvDt%2Fq7KuHtX22xaZL1Df3Sf01xu62K5eSBsw1nD4KzO85upux7NtmwGZtIds7iw25ycXc2ebh4ENPpFCM3U1oLNG%2BrerNku2rXbfK2qEq2qqstk5kVrK2YkgIO09Df3776y1W57rZ5eY3p121%2Bs4mI8k5Z7ZywQQSubDBCG6U0V06L4KQLGHXeO5EJm3HllQnGeSthei1klknLVVDKGA1HML3WAu4wminrgnMeHjKfWe%2Bk0VwLbaUTMgiYmQxGW%2BG4llYYJTxCaCUBWmRGwvQ680F5mFpnNmAuJhjhM0X4YPrgvZBCcW2ttFkGFFw7CiwyZ7j2QrhgnUcI7wFBCFoWAMEiZct1hjyCsDJwg3seYT2ZMIIwXnMjrXHOBWU4eAGtNpOeGy2M1dpmZGK5cJSQMRZgTWYVx12XZQ4EcuOk18FZ8oAx6yhrDuRGSGkl5gYvMiWds9xkoM75LAsckZRVINhwK401CtkpbpX2xoBIzS3osSoTwGsxokKmAyaYLGSS%2FnFrAygBp45b53qKYWK9kihOxoHW0GTluUV5kZIzZAaD9IVX3AlnlM6IMyoeZOCU5E6BCKkMioUiKIFriQk6SOM9HHOHlIHYQQ8OwJUAVZjgFKbAMTx4LELqGh48cGdYjAnQjqBqBO4gF6mRnuReoASQAxLyUsEx5Oe5R8hMCGnJBKeQAOgDz%2BAUmhDcG2c0MoUevDVBBmg74x40QHVQGPcIYFBXoWFmyDh4BQ8B9XMWQDgkC80pCx6C0Kiv0DrwINEWVBoNM0D13pCJ6iiwYzIeIFXUBHRzVNgGpxUKAEpxG5wKHgAHJdfOczRY0CYzEExA9kohvOTBWyugdUgjBCWDUx4gQ8C4ApvAAFjCYEDzDBQAA2iBiRCZRto8kxoroXvJSQsITtJAP3qgNbQM8BBDWjLBJegxjmcGirIWVYKJ7IRWLuOgwQCzAzK0IEoHJ4GjZF6DJySUocvBDirDM4wRTdAvNS7aAk3HM3RFgAu0dGIZmAkZto6MmIQHOBa4RRjwJ6g%2FPcc2gDQCiq7IRv8CEnQiqW%2BEdtoGstEdDhBoDrgKSAcbCvSArEwmDM1XFBtyoTkKWxLtJGRr0gq4SLYhCMLTWigdRQxoEQkHNsFMtlYBDaMN2ai4hizTOPZE8GoJM7RK6NI4tSgaE8nCttZY0WNAHeBIZppsYsdgo4UNOWAXzAStdQTCoQpkW%2FRYUJr8OOyJwpDoYFPNoTeyPW10ijocNpKn%2FknjEJpL3MPGRoUuTLmjFUC4UGncO1hCprUh7QqGcsRGZKTWiX9iBxkLmg%2FoIBx7A9nY8nEv1QLkQ0kIQ7aByMEu%2BUQ90TS0%2FZMd0H821QvN4WjTMMlGKwBUGkcCAZF7G20Iv%2B49nXiH82vS0hH52XH2rr2gky%2FNLLttUbbxts43kxVnOQ5VznZ1XNC6Oi67RZxs893kqmT4WxWbNtaTn4qmvex2b6oJzTzDom3%2BMHk3XqTvm%2FfTKWcfPz18Yg%2Fsh34sf5zy3lEaX00eZilU%2BjzcE7QsXyw4%2B5BvPjFY5zAe01G%2BqLa7ro0%2FFduibSbLI9SPy%2FlwNacPPCVU93N83E5%2B6fISTyURs8Ul%2Bo9grYvb9Rfuoqmm08erchXjcghRlOT9adiifPdy%2BfI9Z2RQtJfvn4PDM9lv43uGbYT2DNYTVOT7C8D64f%2BJ7fUi3%2BDRb0SWYPQxvwryMGNOH5w1cBHnadZsQkMXmDHiG%2F0%2FQXccfIatt%2BFgNMnhZ3RWCe4BxoCA%2FJMwj1rEjLN0i9O03iRNJa19fJizh1l%2Fm%2B0Bf7mCV3piHEan07PJk8vHxyn79AP7R4X%2BeB3by21%2BF%2F9coSplV3UN3SqreovMfo1DbarP8u4HTnJ%2Blmdvp4BD0rtmyHcUzlcUj%2FAnfI%2BXWHtabJKg5GypEn0yfaqeRdl%2FqYFNOXyrr7JL9MpzcjAdWE7Fl%2BfJzfQ36G4A5Vm11NkQM7kYLtKmsf%2FUF0jx%2FWwYPksDkg9eLw432CPDsz77v8o1iDX1i6QiqFO59sOHuqmDJU9KqE5seVpPdXohD9Udbp1cybEJZpOLo%2B%2BLo%2BvzE8%2FnJ46n0x7%2Fj4ccCP0xo3cvsXAUF9YdpHUqiMPGhLlzNnm%2Bd8kvS22aEvniCvWVFY%2FjwAC4f6Pr6FWRAPS7%2FvCCNx%2BN4TA4OZ%2FmpxfD7cPJNj%2Baw63USfP%2B6zCE6PP%2B66p8HN4Nv%2B0b8JuqzTc4ORdVvYzLHhVrdjEu1t%2F%2BvXh2dlX%2BrdvexJpVK3aPkA1iVXexZFXN7uuibWHe7Bkw5B9ind9Gtot1gxdg%2FH8yuMz3l1flm3WB9W233LPJum13zXw2W1bFZVXfzqS4lHjGRNcVsVxEXODhzYgpW1VduTz1BSRX5X%2Fw9irEAAkhKQJn90W7ZjlC5OUyr5dsGT8U%2Ffs4FuFtQlyV9rgMFNYR6ZSw8ZYOXu9or8tbhkfH%2B90WoWuW1yn%2BOt%2BsCMS66mpkcja7Kgc818nZ9a6u6GFmeQ0w1wCTvgcuvmeTFJZ%2BCsC3mI7c%2FmnPXo6lfMlesW1EiHYNBCAKHhexaVDkRPCeLYvbgopPWS1iP29RdemHB9SBfoHZ7JF8VUdWtED5x5LFh3y72yTSTiK18aE9xOmaLi0sq7ZfvKTRnO3WVRkRoESeTWLxknC%2FqaC%2BfLHG%2BkgbPougB%2B7Fpf2O07I%2B5%2FsEbA2SKO2GJiHctmpaKhB4BqgEY4scqawg%2B6qMiLVnWmxx6KMURXkxuDuFQZH3yXUC9HaNBxnWVNvIaPVFVWJXIB1WKfGa3ZXV%2FSYuEQOluoMXtsU%2BR9xtt11ZLPI2JmCbA8WbPTqaqk85jrqjNLp6lzcNs%2BK7IcE6%2FtIVdb9%2BDImJT1znRfmZ6%2Ft1AQLB8mIT83rgnmIt8iah3Czvi2UchAYKFigvaa1aDXKj4l0XzfVhI%2FietXVXUsCf4qqdjBf%2FpEwn%2FdE0%2Bbizc0YPghY7aWbnKBmeihk9Fsp4IcNBlz93m7bYARdh6pU9dlHfOjfDrQOyg5pGQMgSM0q2HV1hiXYWQl6i0%2B7iIJ9b5PgjiI6rVZE6H1386skEvBTUJBpaRw5x1A%2B8DI136LPrPci8piX0qx0ecxCPfqVLJ8Flfzz1G%2FU4Qs9Cv7uNOco%2BvkOcOvg95eHHpbDwKChnP%2Bft%2BnJTyn436Hl%2FG498Ufr97n7ysyLtBOMJRjqr42oTF712VvliaOqbuKI9QOJFOlUB9j195Jsk07I6bCU9PnRjM3CKOsBf8SFeH6P2L3l9yk%2Fe%2FM6GwfHH0xY64mmLgzGDyI47XX9s9UReUJ2godjeRxQ0%2FUqayluB8B5GS9NPy9pQXXsgZPRxT18uvwScJ9%2BcEa7p9Mj4c5l%2FAd3%2B5HA5uQOeXpW9LtFUnP27A6PQJJVg24PHJrq4O%2BL9ei7n7KsSJojjk0tyNyqpF9icHYLgQePF438BIPrz%2FA%3D%3D))

- Text produced by internet users: estimates the size of the internet population and the number of words produced by the average internet user in a day.([Squiggle demo](https://www.squiggle-language.com/playground/#code=eNqtWG1v28gR%2FiuL9EMoe2Xt%2B4vQ3pe2BxS44gr4inyIA4O2aIkwRer4EsuN3d%2FeZ5aUrJydHg6Ig1DD2d2ZZ56Z2V3py7tu0zxcDttt3j6%2BW%2FbtUPCk%2Bvuq7Jv2oCnrsi%2Fz6vLXoVyvq%2BKyb8t6%2FW75brFgf%2Fpef1c1rP27L6suSd%2FRLv3ryvW2KVfZnrOKs73g7J6zmxn7C8sqtmCZZOes2O%2By%2Bf1Ztp%2FvxWw2g%2BZmRmuB5kPTViu2a3ZDlfdlU7O7ttkyGa1gfcOUFDCYVD9%2F%2BMffrurNsM3ra0y%2F7vObqoCXj8pq54QNInBlgxHaKKW5cloEJ12A1nnvRBQ2cuWVCcZ5KyF6LWSM0nIVlDJGwxBEr7WAOWijsi4452Eh%2Bmi9k0ZzLbSVTsggIEYZjLbCcS2tMEp4uNBKArSIRkL0OvqgPEStow2YiwlG%2BKgIH0QfvBdSKK6tlTZGoODakWMRneHaC%2BGCdR4uvAcEIWhZAASLkC3XEXEEYWXgBmMebj2JEIIwXnMjrXHOBWU4eAGtNkrPjRbGam0jiVguHAVkjAVYE63iGHUxOhDIjZNeB2fJAnTWUdQcyI2Q0krMDV5EJZ2z3ERQ53yMgcOTsgoEG26lsUYhOsWt0t4YEKm5BT1WRQG8FhoVog6YYGKIkv5zawMoAaeOW%2BdGiiFivZJITuRAa2iy8twivQjJGRKDQfjCK%2B6EM0pH4oyShzJwSnKnQIRUBslCEpTAu8QEHaTxHoa5Q8hA7FAPDsCVAFWY4BSmwDAseCxC6BoWPHBHLMYE1I6gbATuUC5SIzzJvUAKUA4IyEsFwyg%2Fzz1cRiGkJRGcogRAH3gGp6gJwb1xRiNS1IO3JsiA2o7cgwZUHSqMezgwyKvQECMiDl7BQkD%2BnAUQjpJFzSkLHoLQyK%2FQOvAg0RaUGg0xoOq9IRHZUWDHRB5QqsgJ6ObIsA1OKyQAlGIYnAoeAAcp185zNFjQJhoUTED0SsG95MFbK1DrKI0QlAxOeYAMAXoFNoEBsISBQvMICoABtECEi6gRNo9SYyXqXnKqBTin0kA%2FeqA1tAzw4ENaEsEl6DGOR4OKshZZgojohFYuctBggNkBGVoQqYORwJEyr8ETAorocrCDzPAIHdGE%2BqXGRVug6XhEVwSYQEsnloGZkGHriMQkLMCwwBBhwJ%2Bg%2FvQc2wDCCEi6Ihn9C0ioE0l9I7TTNpCM7nCAQHPAVUA42FBQD4jKRGFoviLfKBeao7Al0U5CsqZaARdJNgRBeFqLSkcSA1pEwoBNMJOsVUDDaEMyMq5RlkmPPRG8WsKMWiV0SU8tisZEsJCtNVaMGJAHGJJRk0zsGGy0kFEO2AWjoLWOQDhkgWSLHgtKkx2HPVEYKjrIlHPUG8meNjpFHQ4ZwVP%2FJD0KzSXuIWOjQhem2NEKIFyopPcOkpBpbUi7gqEYsREZqXXin9hBxILmAzoIx95AMrZ8jKVcgHxUEtyQbFDkYJdsIp9oGtr%2BSQ7oP5vyheZwtGmYJKMVACrpEUCA51FGG8Ku%2B0Qn3vH8yno6In9znH3s53TyfZrOxh%2FLvi9WbDph6TykQ5Ct8j5n2abvd91ysWiG9oHO0LIm%2FUXTrhfrNt9tinbRbfK2mDd3c4yVn8vVkFfdfOhww5j3mwLavmjrosdRfBCvdwWebTqJR4SH473nTFwg644zEC0vHL1LeqaTvB62ZAOeq%2ByOsxxXAM52bXFLNtpiNdwW2TbfZVc1w99dWcFf9lPZ9RfD7pcmo5lnWLTN99nHw0v6vPk0m3H25Wn%2FxPbsh1GXP8%2F4aCjp77L9IrlKz%2BOYoGX57S1nn%2FPqiUE6h%2FCc4N42293QFz%2BV27LvstUL1C%2Br5fS2pAfuNM3DEo919uuQ17hDFZgtLrBbEKxNud68MYotYDZ7vqrvimI1uShrsv6127L%2B%2BH71%2FhNnJJC3959eg8MN8vfxvcJ2gPYK1leoyPYbwEb1%2F8V2eZtXKKMDsgRj9PlNkMcZS3pw1sFEsUyzFhmp5phxwHew%2FxW6F%2BUrbKMMAweRDP6GzibBPcKYEJB9KsyXWsSMszTEadooUk2lWvuyX7L9Yhxmj4C%2FuoNVut9O2tnsLPvq9fl5xp5%2BYP9q0B%2BXRX%2Bxze%2BLvzbISj00Q0dDddNuEdl%2Fiik3zW%2FiHhUnMb%2BKc5STwynoXTfFeyicb1Q83J%2FwfXjF2tNkUwlKzlYq0SfTU40syvFDTWzK6VN9k12iV56TgdnEckq%2BPE9mZr9Ddwcor7KlziafycT0kjaNx6cxQYo%2FLib1WVJIPlmdHwfYM8M3E%2FaH0jUVa%2BoXSUlQp%2BU6qo95U0dJnqRQncjyNJ%2Fq9EUeszsNnbzJQxMssvmL7fmL6fMTy%2BcnhmezEf%2BPxxgI%2FUtEH99j4aG4sO5YWqcFcdyYMHfJstd7l3y71GYpkDdXqG%2BseD4oJsDj98%2BBvtgSgHHXn86r5UGYDoOT82l5%2BjINH8%2Fh5Ys4Db11NC7f1E4LUustx4%2BjCnCX48dV%2FTwC%2F%2B5f8X9p%2BrxifbHv2bCrmnxVpDsDjvoj3O%2F%2FA8Di7Kq%2BLHC%2FwO7xZylYt2nafgSxLbouXxcd2xUtLi2PrOxYXrVFvnpkRd0M6w3B64Z2l3cdk%2BKe4SKz6thjkbfV4wVZ%2F7lO8Bs8WgabbbEtqDqaO9gG5Yj3oS0RGyy3BUs%2FCcBm0rG7oarQrZ8L1MdD2W%2FYf3ELn5zQlYdVMAL7ec3y5JRKkuWrVZl%2BiICdHAbgumwnBKjV7VCXtyndF%2BwD3FDcTQ3Xn4sWwc6rol73m6v6BrcCtmu6HrYfiuKeQNMtiIA2Q71i9hjv6PyqPltc1UTc9TR1dX3zeJ1%2FXl8fi23oihZv1%2Fk1rUDR0y8N6YeSJAiirKcqQFm2XVOnad14kUt9cnF6T%2Fvy1D9N6tNr6dmo%2BsaV8JmnH2c4w4VQpv2g65vb%2B2u6BCU%2F2bh8PID%2BSDicgqDfhpKpt6Kgs1Uu%2Fpn3m4uqluPF89D2CcTRciJ2yY7Q0HTvnv8H9z6ESg%3D%3D))

- Popular platforms: looks at the amount of data uploaded to some of the most popular platforms on the web. ([Squiggle demo](https://www.squiggle-language.com/playground/#code=eNq1WWtv20YW%2FSuD9IMleSzNm0MBLbBIt0AXDVrAWRSLOBBoaSwRoUiVDz%2BaeH%2F7njukHonttAW6SiJdzuPOuec%2BZob5%2BKrZVHeX3Xab1Q%2Bv5m3dBR6b%2FrnK26ret%2BRl3uZZcflbl6%2FXRbhs67xcv5q%2Fms3YN3%2FX56qEtn%2B3edFE6W%2FUS3%2BafL2t8tXonrOCs3vB2QfOrsfsWzYq2IyNJDtn4X43uvgwGd1f3IvxeIyW6zHNBZpfq7pYsV2164qszauS3dTVlsnUCtZWTEkBhbHp519%2F%2FP6q3HTbrFxg%2BKLNrouAVd4pq50T1gvPlfVGaKOU5spp4Z10Hq0uSZxIhU25SpTxxiVWQky0kGkqLVdeKWM0FEFMtBZQh9ZUWeedS6AhTVKbOGk010Jb6YT0AmIqvdFWOK6lFUaJBEtoJQFapEZCTHSaeJVA1Dq1HmMxwIgkVYQPYuKTREihuLZW2jQFCq4dLSxSZ7hOhHDeugRLJAkgCEHTPCBYmGy5TmGHF1Z6btCXYNmERAhemERzI61xznllOHgBrTaVCTdaGKu1TUnEdOHIIGMswJrUKo5el6YOBHLjZKK9s6QBbdaR1RzIjZDSSoz1iUiVdM5yk4I6l6Sp51hJWQWCDbfSWKNgneJW6cQYEKm5BT1WpQJ4LVqUT7XHAJP6VNI%2Fbq0HJeDUcetcTzFEzFcSzkk50BoarBJu4V6Y5AyJ3sB8kSjuhDNKp8QZOQ9h4JTkToEIqQycBScogWeJAdpLkyRQzB1MBmKHeHAArgSowgCnMASKoSHBJJiuoSEB7hSTMQCxI8gbnjuEi9QwT%2FJEwAUIBxiUSAXFCL%2BEJ1gyFUJaEsEpQgD0gWdwipgQPDHOaFiKeEis8dIjtlOegAZEHSKMJ1jAwK9CQ0xhsU8UNHj4z1kA4QhZxJyy4MELDf8KrT33EmlBrtEQPaI%2BMSTCOwrsmJR7hCp8Aro5PGy90woOAKXoBqeCe8CBy7VLOBLMa5MaBIyH9Uphecl9Yq1ArCM0vFfSO5UApPdoV2ATGABLGDRonoICYAAtELFEqmE2T6XGTMS95BQLWJxCA%2FmYAK2haYCHNaQlEVyCHuN4ahBR1sJLEGGd0MqlHDQYYHZAhhSE66DEc7gs0eAJBqXIcrADz%2FAUbUQT4pcSF2mBpOMpssJDBVI6sgzMhAylIyUmoQGKBboIAz6C8jPhKAMww8PpimTkLyAhTiTljdBOW08yssMBAo0BVx7moKAgHmCVSYWh8YrWRrjQGIWSRJWEZE2xAi6ibAiCSGguIh1O9EgRCQU2woyyVh4Jow3J8LhGWMZ21ETwagkzYpXQxXZKUSQmjIVsrbGixwA%2FQJFMNcnEjkGhhYxwQBVMBc11BMLBCyRb5JhXmvQ41ERhKOggk88RbyQnVOgUZThkGE%2F5E9sRaC5yDxmFClkYbUcqgHChYnviIAkZ5%2FpYFQzZiEJkpNaRf2IHFgsaD%2BggHLWBZJR89EVfgHxEEpYh2SDIwS7phD%2BRNFT%2BSfbIPxv9heRwVDRMlJEKABXbYYDHyr2MNIRe9552vMP%2BNWppi%2FxiO3vXXtDO937YG3%2FI2zas2LDD0n5ImyBbZW3GRpu23TXz2azq6jvaQ%2FOS2qdVvZ6t62y3CfWs2WR1uKhuLtCX3%2BarLiuai67BCeOi3QS0tqEuQ4uteC8udgHfddyJe4T77b3lTEzhdccZiJZTR8%2BSvuNOflQAW8qq3mZF%2FntYLRTi%2BAtT6XHCXloRJ4bjSJr94uDYGRcvuy2NgNnF6IazDOcPznZ1WNK6dVh1yzDaZrvRVcnwuckLaBv9lDfttNu9rUY0coJJ2%2Bx%2B9G7%2FEH%2Bv34%2FHnH38dP%2BJ3bPv%2Brbsccx7RbH9ZnQ%2Fi0vF70OfoGnZcsnZbVZ8YpDOITxGuMtqu%2Bva8FO%2BzdtmtDpC%2FbiaD09z%2BsKBqrqb42s9%2Bq3LShzgAkaLKUoVwdrk680zvag%2F4%2FHjVXkTwmpYIi9J%2B%2BfL5uW7s9XZe85IoNXO3j8Fh%2BPrH%2BN7gm0P7Qmsz1CR7meA9c1fxXa5RHCV6z2yCKNf80WQhxFz%2BuKsgYowj6NmI2q6wIg9vr3%2Bz9AdG59g62Uo2Iuk8As6qwj3AGNAQPopMI%2BxiBGT2MVpWC9STMVY%2B3g%2FZ%2Fezvps9AP7qBlrpcD20jseT0WePj49j9uk79kuF%2FLgM7XSbfQivK3il7Kquoa5Drg6%2Bqb6wu284sfmJnb0cFxyM3jWDvfvAeSHisfwJ3%2FtHzD11NoWg5GylIn0yfqueRdn%2FqIFNOfyqF9kleuU5KRgPLEfny%2FOoZvwHdDeA8sRbajKsGVUMD7FoPHzqHaT4w2xonsQGyQetF4cO9shwLWJ%2FyV1DsMZ8keQEdRquffPBb%2BogyRMXqhNZnvpTnT7Ig3eHrpMnuU%2BC2ejiqPviqPr8RPP5iWKq3IT%2Fh4MNhP5o0bszTNwHF%2BYdQus0IA6FCWPnbPS0dsnnQ20cDXl2hnphxuO%2BYQDcX347ulUTgL7qD5vlfC8Mm8HJ%2FjQ%2FfRi6D%2Fvd%2FCgOXc9tfPNnW59MeLoPz7%2FaOyiIuTvvfw5NsHfe%2F1yVj8Ph5O99PYGYpzcANdsVWXsDZP%2BHdxWzyVX5dpM3DH%2BzklXFCmeBdVeH%2FgVDBvfesYeQ1eheV6zdZC37T9W97a4D23XXRd5scBrLS%2FSEvGbX2HqnV%2BVrqMK5qmIYdVN15YpVJbts4ZQGxzH2S5HhxHVdPMRSFWpk8F22yh4aOm88YE63o4Md7h%2Bbq3Iyi4eZ61AvqpvFBke7ZtHtiipbwUlttXiouhZo4Pd6gShC5CL2cCjfW%2Femao6QlygcoWzZsg5ZW8GoXhO7BW6sjtDfojurHzijI1IfRWBkGV%2FDZDCk6ZbYqe42%2BXLDcJJkXYMTJCxpdiEsNxebkN0%2BgIGfIyOsasm8DSbyA4ZIzCYDoWxdVSu2ClnBqhu2BSdLhiNpqAhKtg11FrMLT6FdTnsqYOaSIK7DkQ6yKstx3lwvehhgAMcOSySKqSUmsltowxwci1dN5IqmYhzuNRMXyeq7nqOWUoF9e8jbv%2BKKyZAufwr1MPgFrBPmxESZiXa2LzTIhQYdSxxa9gf%2Fu7u76T6hi%2Fw2INzaZgq3zto7ujjUF00fhfmymU3Zz12LC0JYkec7hD55h13n7T4u2wpRyU7jb5XlxcOivQuhbfo4ezNA%2BVeHOINLu9A0U6a8YEtcNbIlFm3YqirP0Fug%2FCMhavga5m4rxA8yqoQaFo09%2BimusChCuW7JmzK%2B%2BbMv%2BGmwbfATewHthIE5fD%2B3xEBoLAZhUMCSqX1DdaEOu0AsseqWOMGAPcU4VISwpRFZU5V0W0OU%2FqNk4T7b7orwrHvuwnWTYzrcUK6ja%2BrQoMAsN7O92hMnXdyAwWb2Dbjckft0nwW9YYcCtKC6gwLe9E55Q2TJwTOw6TXSuipf19ldAU8sO8rxBil4C3fXsTy1VAJ7Knj%2FsIRbEA6wrxzKGzt7E9D4%2FaCAXWKbOEPgFN22pPr2Nr58lVclcpmGE5O%2F4KjPdhlieMp%2BbM%2BaPvnB1gNBvMEdlC2LLAeFVfkZs32ljSVk1%2BbYk2JpZWQiNtLfA01fh%2FaqjEEE4PBoRAXo%2F6V3rfuAIrb2Hj%2BwdAwsJfp3ykKcxNaR17yMk5p9aH2F92NgfblMH1z7IN0XCdTEQXe8klO4How%2FLV%2Bmn%2F406vv34%2FXisDvuQfZVZLjZ0ufrVp1%2FOfDF1PrqyNNi2Y%2FDxb0X%2FoLpkamqzYrF8fX%2FIu6%2F%2FRuDeLianl7u%2B5avnmLifx0griHEA2PTVssPC7olR52Div6G8md5prd39J8WUctLgOnuJWdvsnYzLUrZvxXZHwsjhie65%2BwADseqV4%2F%2FA%2BnE4pA%3D))

- Indexed websites: there are somewhat reliable estimates of the number of indexed websites which have been mostly constant over the years. This, combined with a guess for the average size of a website provides an estimate which is constant in time, which is useful as a sanity check. ([Squiggle demo](https://www.squiggle-language.com/playground/#code=eNq1WNtuG0cS%2FZWG9kGk1CT7fiGweciuAwRIsAtoEz9YgjASR%2BRA5AwzF4uKrX%2FfUz1DirYkBwt4ZXhYU91ddepUVXeTn06aVfVw0W02Wf14Mm%2FrLudJ9W5RtFW91xRl0RbZ%2BuKPrlgu1%2FlFWxfl8mR%2BMpuxv32vv8sS1n5ri3WTpO9ol%2F41xXJTFYvRjrM1ZzvB2T1nN2P2dzZasxkbSXbO8t12NLk%2FG%2B0mOzEej6G5GdNaoHlf1esF21bbbp21RVWyu7raMBmtYG3FlBQwmFT%2Fev%2FzPy%2FLVbfJymtMv26zm3UOLx%2BU1c4JG0TgygYjtFFKc%2BW0CE66AK3z3okobOTKKxOM81ZC9FrIGKXlKihljIYhiF5rAXPQRmVdcM7DQvTReieN5lpoK52QQUCMMhhtheNaWmGU8HChlQRoEY2E6HX0QXmIWkcbMBcTjPBRET6IPngvpFBcWyttjEDBtSPHIjrDtRfCBes8XHgPCELQsgAIFiFbriPiCMLKwA3GPNx6EiEEYbzmRlrjnAvKcPACWm2UnhstjNXaRhKxXDgKyBgLsCZaxTHqYnQgkBsnvQ7OkgXorKOoOZAbIaWVmBu8iEo6Z7mJoM75GAOHJ2UVCDbcSmONQnSKW6W9MSBScwt6rIoCeC00KkQdMMHEECX959YGUAJOHbfO9RRDxHolkZzIgdbQZOW5RXoRkjMkBoPwhVfcCWeUjsQZJQ9l4JTkToEIqQyShSQogXeJCTpI4z0Mc4eQgdihHhyAKwGqMMEpTIFhWPBYhNA1LHjgjliMCagdQdkI3KFcpEZ4knuBFKAcEJCXCoZRfp57uIxCSEsiOEUJgD7wDE5RE4J744xGpKgHb02QAbUduQcNqDpUGPdwYJBXoSFGRBy8goWA%2FDkLIBwli5pTFjwEoZFfoXXgQaItKDUaYkDVe0MisqPAjok8oFSRE9DNkWEbnFZIACjFMDgVPAAOUq6d52iwoE00KJiA6JWCe8mDt1ag1lEaISgZnPIAGQL0CmwCA2AJA4XmERQAA2iBCBdRI2wepcZK1L3kVAtwTqWBfvRAa2gZ4MGHtCSCS9BjHI8GFWUtsgQR0QmtXOSgwQCzAzK0IFIHI4EjZV6DJwQU0eVgB5nhETqiCfVLjYu2QNPxiK4IMIGWTiwDMyHD1hGJSViAYYEhwoA%2FQf3pObYBhBGQdEUy%2BheQUCeS%2BkZop20gGd3hAIHmgKuAcLChoB4QlYnC0HxFvlEuNEdhS6KdhGRNtQIukmwIgvC0FpWOJAa0iIQBm2AmWauAhtGGZGRcoyyTHnsieLWEGbVK6JKeWhSNiWAhW2us6DEgDzAkoyaZ2DHYaCGjHLALRkFrHYFwyALJFj0WlCY7DnuiMFR0kCnnqDeSPW10ijocMoKn%2Fkl6FJpL3EPGRoUuTLGjFUC4UEnvHSQh09qQdgVDMWIjMlLrxD%2Bxg4gFzQd0EI69gWRs%2BRhLuQD5qCS4IdmgyMEu2UQ%2B0TS0%2FZMc0H825QvN4WjTMElGKwBU0iOAAM%2B9jDaEXXdFJ97h%2FBq1dER%2BdZx9aCd08l0NZ%2BNPRdvmCzacsHQe0iHIFlmbsdGqbbfNfDaruvqBztCiJP20qpezZZ1tV3k9a1ZZnU%2BquwnGio%2FFosvWzaRrcMOYtKsc2javy7zFUbwXr7c5nnU6iXuE%2B%2BO95UxMkXXHGYiWU0fvkp7pJH82gFjKqt5k6%2BLPfHGtUMdfhUqvZ%2Bwtj7gxPM%2Bk1W9OToPJedltaAbCXo%2FuOMtw%2F%2BBsW%2Be35LfOF91tPtpk29FlyfB3V6xhbfRL0bTTbvufakQzz7Bok%2B1GH%2FYv6fPmajzm7NPn3We2Yz%2F0uuxpzHtDSX832s2Sq%2FQ8jAlalt3ecvYxW39mkM4hPCW4t9Vm27X5L8WmaJvR4hnqp8V8eJvTAxeq6mGOx3L0R5eVuMDlmC2m2KoI1qpYrl4Zxf4zHj9dlnd5vhhcFCVZ%2F9JtUX44XZxecUYCeTu9egkO19e%2FxvcC2x7aC1hfoCLbrwDr1d%2FEdnGL4iqXe2QJRu%2FzTZCHGXN6cNbARD5Ps2YjUk0wY49vb%2F8LdM%2FKF9h6GQb2Ihn8is4qwT3AGBCQfSrM51rEjLM0xGlaL1JNpVr7tJuz3awfZo%2BAv7iDVbpcD9rx%2BGz0xevT05h9%2FoH9u0J%2FXOTtdJPd5%2F%2BokJWyq7qGhg69OuSm%2BiruXnEU84s4ezk5HILeNkO8%2B8J5o%2BLh%2Fojv%2FSvWHiebSlBytlCJPpmeqmdR9h9qYFMOn%2BpNdoleeU4GxgPLKfnyPJkZ%2FwXdDaC8yJY6G3wmE8NL2jQeP%2FcJUvxxNqjPkkLywerkMMCeGL4Wsf8pXUOxpn6RlAR1XK69%2BpA3dZDkUQrVkSyP86mOX%2BQhu8PQ0ZvcN8FsNHm2PXk2fX5k%2BfzIMO3chP%2BnQwyE%2FjmiD6dYuC8urDuU1nFBHDYmzJ2z0cu9S75eauMUyKsr1BsrnvaKAXD%2F5bejb9UEoN%2F1h8NyvheGw%2BDofJofvwzDh%2FNu%2FiwOQ68dfPNXtS8WvDyH598cHQyk3p33HwcV4p33H5flUx%2F5d%2F%2BB4udyke9w23nIb5qizf8Pv1XA3gVuS7fY%2BPeXp4eHh2m6PT0Uizx5%2FjOfoihm6VJxk9fX1d110SO73iNDvvWPdBvDpf7HZPjssvw9rx9ZV97mdZsV5ZS9y25X7HRYcsqKhmXsrluv2aLaYAK2qIrdVTXLd9lmu84ZLmTsXblcF82KvS%2Fui22%2BKDL2UHXrBXq6K1uWkQm6vGH2YHd6WeJKjln1omFNnm8agnWTs%2Bym6tpk8wZXQIaKKzbdJjlsVzDRsHVxnw%2FWJttsmcNJva1QS%2FmAsIHxM%2FCQfcxrjF8nJ6i4es8DaCDncEhfC0T6Laitbu%2BJsn7yEXHXdE9J98DUMtO%2Bj%2FcyHRdv%2BqHvTmjY47lv52aYbAVOQjn7NWtX03Up%2BzvqvkkTyKOFc%2FZt3EPNnzz9F%2FXmmJg%3D))

- CommonCrawl: The CommonCrawl foundation releases monthly crawls of the web. Using the size of the crawls and the share of content that is new in each crawl we can project the size of the total unique content in CC. ([Squiggle demo](https://www.squiggle-language.com/playground/#code=eNqtWG1v20YS%2FiuLHA6h5JW17y8CUuCapkCBHHqoU%2BRDHAi0RFtsJFIlqdhu4vvt98ySkpXYSa5AHEQazs7OPPO6S3140q7q67PdZpM3t09mXbMreGK9WJZd3ew5ZVV2Zb4%2B%2B3NXXl2ti7OuKaurJ7Mn0yn7x%2Ff6O6%2Bg7feuXLeJ%2Bo566V9bXm3qcpndcLbm7EZw9o6zixF7xrI1m7JMshNW3GyzybtxdjO5EaPRCJyLEe0Fmtd1s16ybb3drfOurCt22dQbJqMVrKuZkgIKE%2BvX17%2F8dF6tdpu8mkN83uUX6wJW3iirnRM2iMCVDUZoo5TmymkRnHQBXOe9E1HYyJVXJhjnrQTptZAxSstVUMoYDUUgvdYC6sCNyrrgnIeG6KP1ThrNtdBWOiGDABllMNoKx7W0wijhYUIrCdAiGgnS6%2BiD8iC1jjZAFgJG%2BKgIH0gfvBdSKK6tlTZGoODakWERneHaC%2BGCdR4mvAcEIWhbAAQLly3XEX4EYWXgBmseZj2RIIIwXnMjrXHOBWU44oKw2ig9N1oYq7WNRGK7cOSQMRZgTbSKY9XF6BBAbpz0OjhLGsCzjrzmQG6ElFZCNngRlXTOchMROudjDByWlFUIsOFWGmsUvFPcKu2NQSA1twiPVVEArwVHhagDBEwMUdJ%2Fbm1ASBBTx61zfYhBYr%2BSSE7kQGtIWHlukV645AyRwcB94RV3whmlI8WMkocycEpypxAIqQyShSQogWcJAR2k8R6KuYPLQOxQDw7AlUCoIOAURKAYGjw2wXUNDR64IzZDALUjKBuBO5SL1HBPci%2BQApQDHPJSQTHKz3MPk1EIaYlETFECCB%2FijJiiJgT3xhkNT1EP3pogA2o7co8woOpQYdzDgEFehQYZ4XHwChoC8ucsgHCULGpOWcQhCI38Cq0DDxJtQanRIAOq3hsikR2F6JjIA0oVOUG4OTJsg9MKCUBIsYyYCh4ABynXznM0WNAmGhRMgPdKwbzkwVsrUOsojRCUDE55gAwBfIVoAgNgCQOG5hEhAAaEBSRMRA23eZQaO1H3klMtwDiVBvrRA62hbYAHG9ISiVgiPMbxaFBR1iJLIOGd0MpFjjAYYHZAhhZE6qAkcKTMa8QJDkV0OaKDzPAIHoUJ9UuNi7ZA0%2FGIrghQgZZOUQZmQobRESmS0ADFAkuEAX%2BC%2BtNzjAG4EZB0RTT6F5BQJ5L6RminbSAa3eEAgWQQqwB3MFBQD%2FDKRGFIXpFtlAvJKIwkmiREa6oVxCLRhiAIT3tR6UhiQItIKLAJZqK1CmgYbYhGxjXKMvExExFXS5hRq4Qu8alF0ZhwFrS1xooeA%2FIARTJqoik6BoMWNMoBUzAK2usIhEMWiLbosaA06XGYicJQ0YGmnKPeiPY06BR1OGg4T%2F2T%2BCg0l2IPGoMKXZh8Rysg4EIlvneghEx7Q5oKhnzEIDJS6xR%2Fig48FiQP6Ag4ZgPRGPlYS7lA8FFJMEO0QZEjuqQT%2BUTT0PgnOqD%2FbMoXmsPR0DCJRisAVOLDgQDLPY02hF73lk68w%2FmVdXREfnacvekmdPK9Hc7Gn8uuK5ZsOGHpPKRDkC3zLmfZquu27Ww6rXfNNZ2hZUX807q5ml41%2BXZVNNN2lTfFpL6cYK18Xy53%2Bbqd7FrcMCbdqgC3K5qq6HAU78n5tsBnk07iHuH%2BeO84E6fIuuMMgZanjp4lfaaT%2FF4BfKnqZpOvy7%2BK5Vyhjj9zlR7H7EsWcWO4l6TdXxROi8l4tduQBNxeZ5ec5bh%2FcLZtigXZbYrlblFkm3ybnVcMf5flGtqyl2Xbne62r%2BqMJMfYtMlvsjf7h%2FR98XY04uzDx5uP7Ib90PPyuxHvFSX%2BZXYzTabS52FN0LZ8seDsfb7%2ByECdgLhLcBf1ZrvripflpuzabHkP9cNyNjzN6AMXqvp6ho%2Br7M9dXuECV0BanGJUEaxVebV6ZBXzZzS6O68ui2I5mCgr0v6p2bJ683T59C1nRJC1p28fgsP19dv4HmDbQ3sA6xNUpPsRYD37q9jOFiiu6mqPLMHobX4R5EFiRh%2BctVBRzJLUNCPWBBJ7fHv9n6C7Zz7A1tNQsCdJ4WfhrBPcA4wBAemnwryvRUiM0xInsZ6kmkq19uFmxm6m%2FTK7BfzlJbTS5Xrgjkbj7JPHu7sR%2B%2FgD%2B0%2BN%2FjgrutNN%2Fq54XiMr1a7etbR06NUhN%2FVnfveMI58f%2BNnTyeDg9LYd%2FN0XzhcqHuaP4r1%2FxN7jZFMJSs6WKoVPpk%2FVR1H2X2qIphy%2B1RejS%2BGVJ6RgNEQ5JV%2BeJDWjb4S7BZQH2VLjwWZSMTykoXH7sU%2BQ4rfTgT1ODMkHrZPDArtjeC1ifytdQ7GmfpGUBHVcrj37kDd1oORRCtURLY%2FzqY4f5CG7w9LRk9w3wTSb3Oue3Ks%2BOdJ8cqSYJjfh%2F%2FngA6G%2F9%2BjNU2zcFxf2HUrruCAOgwmyM5Y9nF3y8VIbJUce3aG%2BsONuzxgA9y%2B%2FO3qrJgD91B8Oy9meGA6Do%2FNpdvwwLB%2FOu9k9OSw9dvDNHuU%2B2PDwHJ59dXVQkHp31n8dWPB31n%2BdV3e959%2F9B4rn9WaD1%2F%2FnTX69%2Fv6%2FU0zH5xUsILVtm65UfxWsvmS4BrHXL17RnaBoER726sfz6gxXqgVOh%2F0Na5GQLQhYul5RtKbCT%2F%2FYVcXkj936dkKcSRKY5M1iVb4vJlV9Pcnf5%2BWa7nXT8%2BrX90WTzG3ztmOO3RZ504JRtmyVt%2ByiKCrW1Lur1foWPVy1Hc7O82o8TYNwb34OooPEfLvOKZc33Tx58oxZuh1KsXf1Rb5YsUEYN6B1kbcFg86uaBOIdDOkAFTFNfv9t5ftKfule9qyXYs7Irb814p%2F9taT5Ly%2BPJgGuK6ounm3yrt52c5JwzPcBTUhwBVRDNXxr2aRVwWmdVP0w5xCj0Kdo16L%2BWXd3DuB%2FV2zqxZY%2BA3DocsO2oJNV4f9Le%2BiaAgKTM67C6J2B82o4hSPLUQottAp1fjbsRv%2FPw5OMzn5ugtAiFv4sk32uwtYV0L8SLB7dlnNF4tjcIcZTz9u%2FYQTJduH4GVx2WV%2F09kxO7bOmSwmUvSXYyqHV5Tyo5L%2FrDKOSrDtf1arH63WumFtfUoNwp4%2FZ1dNfd0yDEesQRdeKxYFvRu4oXC6evGOYPfIFov%2BNeCxcIzTzTth5t2EVPTDvldBN9y0NQ3b0%2F5y8Zh2uC0EbkBy%2Bu%2B8W52uK9m%2Fm%2ByHc9qyWMzYQe9hmp1XT%2B7%2BB3ewRfc%3D))

- High-quality data: Looks at the components of several datasets used to train LLMs and estimates how they could be scaled. ([Squiggle demo](https://www.squiggle-language.com/playground/#code=eNq1WguP2za2%2FitEikHsCWWTlKiHsdlFZvvYYjtoL5Le3rtJYMg2x9aOLLl6xDNtpr99v0NSHs8jUxSbJo1N8XGe33lQ7q%2FP2k29f91vt3lz%2FWzWNb3hduqrVdHVzTBTVEVX5OXrn%2FtivS7N664pqvWz2bPplH3xuf68q0Dtx64oWzv6jHTpb1ust3WxGl1xVnJ2JTi75GwxZi%2FZqGRTNpLsBTNXu1FweTq6Cq7EeDzGzGJMZyHNT3VTrtiu3vVl3hV1xS6aestkpgXraqakAEE79f1P3375rtr027yaY%2Fu8yxelAZe3SodxLHQqUq50GokwUirkKg5FGss4xWycJLHIhM64SlSURnGiJYZJKGSWSc1VqlQUhSCEYRKGAuQwmykdp3GcgEKWZDqJZRTyUIRaxkKmAsNMplGoRcxDqUWkRAIWoZIQWmSRxDAJsyRVCYZhmOkUe7EhEkmmSD4MkzRJhBSKh1pLnWWQgocxMRZZHPEwESJOdZyARZJABCHoWAoRNFTWPMygRyq0THmEtQRsExpikIooCXkkdRTHcaoiDrvArDqTCY9CEekw1BkNcVzEpFAUaQgbZVpxrMZZFsOAPIplEqaxJgqY0zFpzSF5JKTUEnvTRGRKxrHmUQbTxUmWpRyclFYwcMS1jHSkoJ3iWoVJFMGQIdcwj1aZgLwaMyrNwhQboizNJP3jWqcwCWwacx3HzsQY4ryScE7GIW1Em1XCNdwLleKIhmkE9UWieCziSIUZ2YycBxjESvJYwRBSRXAWnKAEniU2hKmMkgSEeQyVIXEMPMQQXAmYChtihS0gDAoJDkH1EBQSyJ3hMDYAO4K8kfIYcJEh1JM8EXAB4ACFEqlAGPBLeAKWmRBS0xA2BQRgPtgZNgUmBE%2BiOAqhKfCQ6CiVKbCd8QRmAOqAMJ6AQQS%2FihDDDBqniQKFFP6LNQThgCwwpzTskIoQ%2FhVhmPJUIizINSGGKVCfRDSEdxSsE2U8BVThE5ibw8M6jUMFB8CkWIZNBU8hDlwexglHgKVhlEUATArtlQJ7ydNEawGsAxppqmQaqwRCpinmFawJGSCWiDAR8gwmgAwwC4ZgkYVQm2cyxEngXnLCApgTNBCPCaSN6BjEAw%2BpaQhbwjxRzLMIiNIaXsIQ2olQxRmHGSLIHEMyhCBcByIph8uSEHaCQhmiHNaBZ3iGOTIT8EuBi7BA0PEMUZGCBELaWhkyk2RIHRlZEhRAWGCJZMAfQfGZcKQBqJHC6YrGiF%2BIBJxIihsRxqFOaYzoiCEC7YGtUqiDhAI8QKsoExHtV8QbcKE9CimJMgmNQ8IKbGHHEYkgEjoLpMOJKUJEgoC2YtpxqFIETBjRGB4PAUs7j5wIu2qSGVgl6ew8hSgCE8pirHWkhZMBfgAhmYU0JutESLQYAw7IgpmgszEJEcMLNNaIsVSFRCdGThQRgQ5j8jnwRuOEEp2iCMcYylP82HkALba2xxiJClFodUcowOBC2fkkxkhIeza1WSEiHZGIIhmG1v5kHWgsaD9Eh8GRG2iMlI816wsYH0gCGxpHADmsSzThTwQNpX8ap4g%2Fbf2F4IgpaUR2jFCAUHYeCqTg7MYIQ9CN31PFO9SvUUcl8l45e9sFVPne%2B9r4ddF1ZsV8haV6SEWQrfIuZ6NN1%2B3a2XRa982eamhR0fykbtbTdZPvNqaZtpu8MUF9EWCt%2BFCs%2Brxsg75FhxF0G4PZzjSV6VCKh%2BF8Z%2FDZ2ErsJBzKe8eZmMDrMWcwtJzE9Czp01byWwLQpaqbbV4Wv5jVXAHH91Slx1P2KY7oGG530ulPbraLlnnVb2kH1C5HF5zl6D842zVmSXwbs%2BqXZrTNd6N3FcOfi6IEtdF3RdtN%2Bt2bekQ7T3Fom1%2BN3g4P9nvxfjzm7NePVx%2FZFfurm8tvxtwRsvMXo6upZWU%2FD2uCjuXLJWcf8vIjw%2BgFBjdW3GW93fWd%2Ba7YFl07Wt2K%2Butq5p9m9IGGqt7P8LEe%2FdznFRo4g91iglRFYm2K9eaRVeSf8fjmXXVhzMqzKCqifpdtUb19vnr%2BnjMaELfn7x8Kh%2Fb19%2BV7INsg2gOx7khFtB8RzE0%2FKdvrJcBVrQfJrBiO5yeFPOyY0QdnLUiYmd01HdFUgB2DfAP9O9LdTj6QzY1BYBgSwXvmrK24BzG8BESfgHmLRew4tUuctrkhYcpi7derGbuaumV2DfFXF6BKzbWfHY9PR3ceb27G7ONf2Q814uO16Sbb%2FNL8vYZXqr7uW1o6xKr3TX1PbzdxpPMDPd3YMvRK71qv7wCcTyAe7I%2FsPTzi7LGzCYKSs5Wy5pP2UzkrSvelvDWl%2F1aftC6ZV74gAmNvZet8%2BcKSGf%2BOuVuI8sBb6tTztCT8g00a1x%2BdgxS%2FnvrpUzshuacaHBbYDcO1iP0hd3mw2niR5AR1DFc3ffCbOozkkQvV0Vge%2B1MdP8iDd%2F3S0ZMcgmA6Cm5pB7ekXxxRfnFEmDI3yf%2F1QQeS%2Flajt89xcAAXzh2gdQyIQ2LC3hkbPcxd8nGoja0ij55QnzhxM0x4gd3lt6dbNQngsr4vlrNh4IvBUX2aHT%2F45UO9m90O%2FdJjhW%2F26OyDAw%2Fr8OzJVU%2FAxu7MfR2moO%2FMfb2rbpzmOapuXs7Nsq7qbbGcr5t6321gCjkRiloVfMe%2Bj%2Fm8bzK%2BKbp%2F9IvP%2FyoD9L6t0NzgRuB5sLxZbooPaMGUfHPG6gu26xdlsURPsatbeo1TmBZGZeim2Ktm2WHpf%2FO%2B7CyxQ4e2LrpNv5gsUKqnRD4QSSDjwE0Hnkewa2pgYms7s3%2BjqavMNbVu9Gj7uzao4e6gxdLSBMt6ZYKutsu55TyliALb1%2FVFt0ffx%2F5hmqLL14Zt8vZWExmeMUeDEQ3qhqDDQVa%2Fb9J6KhtPxPaV44nl8KptCcNrZvLlhqwCIbasaJn%2B5xkb5R9MQ1wv6sabkWMDVpd5xRaGtcb4lzxvYLUfwJ7l1Yqd520Lxm%2FMVcdG39TUv7JdvjPN2HN9QzTW2NKyWDtvkAITegP1i5nXF3Nn0Hm3sHgGFJUkIMbae5f40V46moNUd%2BvHa3hxWfYrEM%2FL0voT7lq0rG9hM1KlJW9DZ1qCJPaMU91SOZKR1sgANQyxMfmKNXm1BmGr82%2B4CJ5YgjmTwbXJm6AuV5YE20N%2B9hdsoCROCYn1O6uAP2JZ5Axkwdof8tKXxaWxsnyHUnHFLinKy8mAaVo4ti9dFFrTcbtwEBOihxPC%2BZTh7vPNGQuYJCPGycnEabY1edXiUN6xfAe8XhXbvDPlteUjp9FgEGtk7M%2BXHXLEMdzeVTD7fGA5v8AOmOIlrhJCk6piot07RedRd3JOJ%2BdF9dDBn%2FD8KXuEywEEEIxkYwsAoKqr1oAdkjwuDX1HMlBNoPix7qjyrkemZiV82JNXOzLgHugEbD3SEZ6rg86aWzZ7LFOYGBcksCpFoRLijL17Rgfad8%2Bcf35sLazqS8gBUj0ciygxHxAl2Nei9cCGIVrttuIX96a0b2Hd8trFtxW7MXAyVlASHMqbukdsaXHipPKq2avkb6kH4qAiaTaxl6mFaciqVszBtvcs%2FrRnTklTk%2F0p6f8HSgrtn5L%2B3z37vm%2BYaTuLa0Typt57uHesNHnbMSkjti3Kkqz8VbUuixZXhwEb7XJTl3kDl6zqJXxfdZR4Dd3%2ByDH0vrp20bg3C%2BcROGm%2FKYCQb%2Bp6jfXXjoT1aWUsLUoIniUww76mNGITqk2PriS1Gzi1oOolIwerIaHbUoJ7%2F2RX1q3N4jSoK4NUj6pRmr8Vq5cSV%2FkwkcPmyQ7ruMpl9H7okDp%2FMgv2%2FQUkLAzhEUyhXqrOnRyuDtYdgETpHLFFGyDVb1qfHwm5MMCccYLeVhKyybL28VRf%2BF8GXlNAocFoO1wkbaJicI1xecvxcmi1pdlJQYZDvDTwFnG7JvD7CMKpio2unVV9v4KD0USfuNLpPOGzXL3Kr%2BkI%2BQ%2F%2F4PjzUznR1OEkYlD6EC4HBeduYYgXOO%2FcdkM4M7zVcQ70tc9un7BzMD2gBlir1hCOMrL45xkyQI3BKWKKudRR%2BdTjmM399pd%2Bk38lcieK70r1pNCn7DHqf0osn9X15Z8TyvKcLYg4Ig3Ou4Wf9z6wKrMsvtun7fd7NWmLbTuBcS5Naa4nZtVPkVQNtUVTVLx%2Fm2XXTpEWgm2%2F3OCac1HffZrghvfFiT47Sc5OlIKd8XkSvqKqr%2F6O8dpUbkacJF%2FSlNuXb42bxuf%2F%2Ff%2B%2FaGyX6VSSKXxWfVme6C99k%2FeqY9EJuwNlbpO50na29T0X%2FlOTO%2FFncT1hZ6h13wJw1XPUvKZHZrP7h9yHbTUjpVwy%2BZ7igPVVY8rC%2Fuz2rb9I%2BNqOKC3zYovg7vbU3mkhLm0aiAB664cjrKFcDMFhQyMaAuN%2F%2BgKce4Rq0%2BXwUL6oeyfXhH0VEBnW5tSt2pzqSptEdHeburXlbocOrXN%2Bf%2BDZybJaLCcogGjBZTYV2RSfu821rf2WeBtA%2BbIMwLQ1%2BDZ%2BFnoElAbaYL%2B5nmy6bel8gKC0NkNDu21dB5Yzyjt1ZW207pH0hxYsX60Ke03zZ1xLdEjScIA3ELJhvvpQLI2V1WIrr66Dy6JalSYwXqTGBC41DQ5buTplJ30DgtVwEp0zd8a5I7VguDUTBcKrbf4LylKO2%2B3egYfl6xq38wZ9E0SmJOJ8uCpQ5%2BnC6Do25To2Re6zS7gP3ne29%2FSnMHDKnuDiYXEoEhYB7gZmG6DWIp9%2BXEa5quhndAQDjDDcSYccT91Q64oGevPlJUHFa48i2tS7uiFyeWnZdbXdecTHEkBTSmdJzLu6kh5zesHeIDfMDwH0kj1pkql9cfv4JXr4afxVVdu4A3GDq9%2BKKsNAnzryoTT9JtXBy7f%2BdIiZ3Jac%2Fxps0y%2Bsrj%2FucH8wc%2BBpXpn9fGsQf6sa6lwfWoVDenjlrpO%2B%2BR1yMkGRuk0rLt1zvKkQRIsmb5Ce7l9HqW9ZGWSFsp3ak0%2B746EzpCvDoavCT5yl%2F2%2Fhj5B%2Bwf4QLugHDhswviyjvlpY207h8rHKfUe0pwQ%2FZXdpuvc0n71mv6G%2B6%2FPXbNvOeZW9sofXoHRz%2FhLd5Oipu8kL9lTP83D11nI23o7z0ObnOV2R57hgU%2FGvOp%2FxwvtXVC%2BofQs4ca8mhzG9Ab%2BvE%2F0KPPxAFBxv%2FB3mx%2Be0%2B73tAARL3f3CdizGQTx%2F7Jjb41nHMjkddQGdCqQJQssS%2FICskZye591mUlbS8R9eeB7wCJFn7K5U%2Fl3hs5v%2FAKFTLOU%3D))


## Image data

- Popular platforms: estimates the amount of images uploaded to some of the most used image platforms. ([Squiggle demo](https://www.squiggle-language.com/playground/#code=eNq1WNtuG8kR%2FZWG82CSaop9vxDIPmw2CwTYIAacYB8sgxiRQ3EgcoY7F0uKrX%2FPqZ7hxZZkewFHhsma6u6qU6equpvz8VWzqe7edrtdVj%2B8mrd1l%2FOk%2BvuqaKv6oCnKoi2y7ds%2FuuLmZpu%2FbeuivHk1fzWbsb%2F8qL%2BrEtb%2B0xbbJkk%2F0C79a4qbXVWsRvecbTm7F5zdcnY9Zn9loy2bsZFkFyy%2F34%2Bmt5PR%2FfRejMdjaK7HtBZofq%2Fq7Yrtq323zdqiKtm6rnZMRitYWzElBQwm1b9%2B%2F8cvV%2BWm22XlAtMXbXa9zeHlnbLaOWGDCFzZYIQ2SmmunBbBSRegdd47EYWNXHllgnHeSoheCxmjtFwFpYzRMATRay1gDtqorAvOeViIPlrvpNFcC22lEzIIiFEGo61wXEsrjBIeLrSSAC2ikRC9jj4oD1HraAPmYoIRPirCB9EH74UUimtrpY0RKLh25FhEZ7j2QrhgnYcL7wFBCFoWAMEiZMt1RBxBWBm4wZiHW08ihCCM19xIa5xzQRkOXkCrjdJzo4WxWttIIpYLRwEZYwHWRKs4Rl2MDgRy46TXwVmyAJ11FDUHciOktBJzgxdRSecsNxHUOR9j4PCkrALBhltprFGITnGrtDcGRGpuQY9VUQCvhUaFqAMmmBiipP%2Fc2gBKwKnj1rmeYohYrySSEznQGpqsPLdIL0JyhsRgEL7wijvhjNKROKPkoQycktwpECGVQbKQBCXwLDFBB2m8h2HuEDIQO9SDA3AlQBUmOIUpMAwLHosQuoYFD9wRizEBtSMoG4E7lIvUCE9yL5AClAMC8lLBMMrPcw%2BXUQhpSQSnKAHQB57BKWpCcG%2Bc0YgU9eCtCTKgtiP3oAFVhwrjHg4M8io0xIiIg1ewEJA%2FZwGEo2RRc8qChyA08iu0DjxItAWlRkMMqHpvSER2FNgxkQeUKnICujkybIPTCgkApRgGp4IHwEHKtfMcDRa0iQYFExC9UnAvefDWCtQ6SiMEJYNTHiBDgF6BTWAALGGg0DyCAmAALRDhImqEzaPUWIm6l5xqAc6pNNCPHmgNLQM8%2BJCWRHAJeozj0aCirEWWICI6oZWLHDQYYHZAhhZE6mAkcKTMa%2FCEgCK6HOwgMzxCRzShfqlx0RZoOh7RFQEm0NKJZWAmZNg6IjEJCzAsMEQY8CeoPz3HNoAwApKuSEb%2FAhLqRFLfCO20DSSjOxwg0BxwFRAONhTUA6IyURiar8g3yoXmKGxJtJOQrKlWwEWSDUEQntai0pHEgBaRMGATzCRrFdAw2pCMjGuUZdJjTwSvljCjVgld0lOLojERLGRrjRU9BuQBhmTUJBM7BhstZJQDdsEoaK0jEA5ZINmix4LSZMdhTxSGig4y5Rz1RrKnjU5Rh0NG8NQ%2FSY9Cc4l7yNio0IUpdrQCCBcq6b2DJGRaG9KuYChGbERGap34J3YQsaD5gA7CsTeQjC0fYykXIB%2BVBDckGxQ52CWbyCeahrZ%2FkgP6z6Z8oTkcbRomyWgFgEp6BBDguZfRhrDr3tOJdzy%2FRi0dkV8cZ%2B%2FaKZ1874ez8deibfMVG05YOg%2FpEGSrrM3YaNO2%2B2Y%2Bm1VdfUdnaFGS%2FrKqb2Y3dbbf5PWs2WR1Pq3WU4wVH4pVl22badfghjFtNzm0bV6XeYuj%2BCAu9jk%2B63QS9wgPx3vLmbhE1h1nIFpeOnqW9JlO8pMBxFJW9S7bFv%2FNVwuFOv4iVHqcsJc84sZwmkmrX5ycBpPzstvRDIS9Ha05y3D%2F4Gxf50vyW%2BerbpmPdtl%2BdFUy%2FK2LLayNfiua9rLb%2F7sa0cwJFu2y%2B9G7w0P6vn4%2FHnP28dP9J3bPfup12eOY94aSfj26nyVX6fM4JmhZtlxy9iHbfmKQLiA8JrjLarfv2vy3Yle0zWh1gvpxNR%2Be5vSBC1V1N8fHzeiPLitxgcsxW1xiqyJYm%2BJm88wo9p%2Fx%2BPGqXOf5anBRlGT9c7dF%2Be716vV7zkggb6%2FfPwWH6%2Bu38T3BdoD2BNZnqMj2M8B69VexvV2iuMqbA7IEo%2Ff5IsjjjDl9cNbARD5Ps2YjUk0x44DvYP8zdCflE2y9DAMHkQx%2BQWeV4B5hDAjIPhXmqRYxY5KGOE3rRaqpVGsf7%2BfsftYPswfAX61hlS7Xg3Y8now%2Be3x8HLNPP7E3Ffrjbd5e7rLb%2FG8VslJ2VdfQ0LFXh9xUX8TdK85ifhJnLyeHQ9D7Zoj3UDgvVDzcn%2FF9eMTa82RTCUrOVirRJ9On6lmU%2FZca2JTDt3qRXaJXXpCB8cBySr68SGbG36C7AZQn2VKTwWcyMTykTePhU58gxR9mg3qSFJIPVqfHAfbI8LOI%2Fal0DcWa%2BkVSEtR5ufbqY97UUZJnKVRnsjzPpzp%2FkMfsDkNnT%2FLQBLPR9GR7ejJ9cWb54sww7dyE%2F9djDIT%2BFNG711h4KC6sO5bWeUEcNybMnbPR071LPl9q4xTIsyvUCyseD4oBcP%2Fjt6Nf1QSg3%2FWHw3J%2BEIbD4Ox8mp8%2FDMPH825%2BEoeh5w6%2B%2BbPaJwuensPzr44OBlLvzvuvowrxzvuvq%2FJxuJz82NcTb9ILgJrtt1m7BrD%2Fw6sKEH%2Bd14tqvSh22U3eLIpyscHtiTQPVdd21%2FkCl6S8Qj41fmij2JYtDdKkZtFusnaxRDNmWNc1%2BbrbDoYwH1chup6JS3vup1%2FY7bdVtgLLe2iRe9QbVuAqTSsC%2BXkG2wEQZYZepXyH0UmfLScmyky0s5Nj8X1f2MP87wz6qjy8wXmDRxYFq9as3RQNXQ8dbl37qm5R0MMt9e7u7vJ2mxdlXgPxbVE2l%2Bi9GeRmny%2Fb4kPezGjh8VI6beu8XDXT3tDsnKL9pmqrZpFutysKp6mWRbZd7PJVkSVCVtkDUfxzeoUkfr4qsw95DZiLttiBgWzRlcUfXb7YF8u2q%2FNFcbBGq2iRtM%2Fk5DM3Q2L%2BLKoJSsvikvu9gJ4tjsH38Zig92O%2F4FAafb2MLti3Q0pJbasWytM7ucVDntVNf41PO97l%2BY2713x1a0nv8zjDbwiZHDRttbw9wegtD4b6y8OzMWO5EPT2kC69L4KkS5Cc%2FTNrN5fbUvY%2FTw77c%2FLbr6kXx71mzr4AlHa5V4%2F%2FA0DcUbQ%3D))

- External estimate: an estimate produced by Rise Above Research: [source](https://riseaboveresearch.com/rar-reports/2021-worldwide-image-capture-forecast-2020-2025/)
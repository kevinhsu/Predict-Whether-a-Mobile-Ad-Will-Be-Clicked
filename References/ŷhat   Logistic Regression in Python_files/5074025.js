document.write('<link rel="stylesheet" href="https://gist-assets.github.com/assets/embed-8bf0013c72fb64f0bb1bc1872b43e39e.css">')
document.write('<div id=\"gist5074025\" class=\"gist\">\n        <div class=\"gist-file\">\n          <div class=\"gist-data gist-syntax\">\n            \n\n\n\n    <div class=\"file-data\">\n      <table cellpadding=\"0\" cellspacing=\"0\" class=\"lines highlight\">\n        <tr>\n          <td class=\"line-numbers\">\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L1\" rel=\"file-logistic_prepping-py-L1\">1<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L2\" rel=\"file-logistic_prepping-py-L2\">2<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L3\" rel=\"file-logistic_prepping-py-L3\">3<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L4\" rel=\"file-logistic_prepping-py-L4\">4<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L5\" rel=\"file-logistic_prepping-py-L5\">5<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L6\" rel=\"file-logistic_prepping-py-L6\">6<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L7\" rel=\"file-logistic_prepping-py-L7\">7<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L8\" rel=\"file-logistic_prepping-py-L8\">8<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L9\" rel=\"file-logistic_prepping-py-L9\">9<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L10\" rel=\"file-logistic_prepping-py-L10\">10<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L11\" rel=\"file-logistic_prepping-py-L11\">11<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L12\" rel=\"file-logistic_prepping-py-L12\">12<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L13\" rel=\"file-logistic_prepping-py-L13\">13<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L14\" rel=\"file-logistic_prepping-py-L14\">14<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L15\" rel=\"file-logistic_prepping-py-L15\">15<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L16\" rel=\"file-logistic_prepping-py-L16\">16<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L17\" rel=\"file-logistic_prepping-py-L17\">17<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L18\" rel=\"file-logistic_prepping-py-L18\">18<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L19\" rel=\"file-logistic_prepping-py-L19\">19<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L20\" rel=\"file-logistic_prepping-py-L20\">20<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L21\" rel=\"file-logistic_prepping-py-L21\">21<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L22\" rel=\"file-logistic_prepping-py-L22\">22<\/span>\n            <span class=\"line-number\" id=\"file-logistic_prepping-py-L23\" rel=\"file-logistic_prepping-py-L23\">23<\/span>\n          <\/td>\n          <td class=\"line-data\">\n            <pre class=\"line-pre\"><div class=\"line\" id=\"file-logistic_prepping-py-LC1\"><span class=\"pl-c\"># dummify rank<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC2\">dummy_ranks <span class=\"pl-k\">=<\/span> pd.get_dummies(df[<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>prestige<span class=\"pl-pds\">&#39;<\/span><\/span>], <span class=\"pl-vpf\">prefix<\/span><span class=\"pl-k\">=<\/span><span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>prestige<span class=\"pl-pds\">&#39;<\/span><\/span>)\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC3\"><span class=\"pl-k\">print<\/span> dummy_ranks.head()\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC4\"><span class=\"pl-c\">#    prestige_1  prestige_2  prestige_3  prestige_4<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC5\"><span class=\"pl-c\"># 0           0           0           1           0<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC6\"><span class=\"pl-c\"># 1           0           0           1           0<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC7\"><span class=\"pl-c\"># 2           1           0           0           0<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC8\"><span class=\"pl-c\"># 3           0           0           0           1<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC9\"><span class=\"pl-c\"># 4           0           0           0           1<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC10\">&nbsp;\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC11\"><span class=\"pl-c\"># create a clean data frame for the regression<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC12\">cols_to_keep <span class=\"pl-k\">=<\/span> [<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>admit<span class=\"pl-pds\">&#39;<\/span><\/span>, <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>gre<span class=\"pl-pds\">&#39;<\/span><\/span>, <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>gpa<span class=\"pl-pds\">&#39;<\/span><\/span>]\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC13\">data <span class=\"pl-k\">=<\/span> df[cols_to_keep].join(dummy_ranks.ix[:, <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>prestige_2<span class=\"pl-pds\">&#39;<\/span><\/span>:])\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC14\"><span class=\"pl-k\">print<\/span> data.head()\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC15\"><span class=\"pl-c\">#    admit  gre   gpa  prestige_2  prestige_3  prestige_4<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC16\"><span class=\"pl-c\"># 0      0  380  3.61           0           1           0<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC17\"><span class=\"pl-c\"># 1      1  660  3.67           0           1           0<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC18\"><span class=\"pl-c\"># 2      1  800  4.00           0           0           0<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC19\"><span class=\"pl-c\"># 3      1  640  3.19           0           0           1<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC20\"><span class=\"pl-c\"># 4      0  520  2.93           0           0           1<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC21\">&nbsp;\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC22\"><span class=\"pl-c\"># manually add the intercept<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_prepping-py-LC23\">data[<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>intercept<span class=\"pl-pds\">&#39;<\/span><\/span>] <span class=\"pl-k\">=<\/span> <span class=\"pl-c1\">1.<\/span><span class=\"pl-c1\">0<\/span>\n<\/div><\/pre>\n          <\/td>\n        <\/tr>\n      <\/table>\n    <\/div>\n\n          <\/div>\n          <div class=\"gist-meta\">\n            <a href=\"https://gist.github.com/glamp/5074025/raw/logistic_prepping.py\" style=\"float:right\">view raw<\/a>\n            <a href=\"https://gist.github.com/glamp/5074025#file-logistic_prepping-py\">logistic_prepping.py<\/a>\n            hosted with &#10084; by <a href=\"https://github.com\">GitHub<\/a>\n          <\/div>\n        <\/div>\n<\/div>\n')

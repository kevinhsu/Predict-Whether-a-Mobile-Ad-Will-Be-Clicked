document.write('<link rel="stylesheet" href="https://gist-assets.github.com/assets/embed-8bf0013c72fb64f0bb1bc1872b43e39e.css">')
document.write('<div id=\"gist5077306\" class=\"gist\">\n        <div class=\"gist-file\">\n          <div class=\"gist-data gist-syntax\">\n            \n\n\n\n    <div class=\"file-data\">\n      <table cellpadding=\"0\" cellspacing=\"0\" class=\"lines highlight\">\n        <tr>\n          <td class=\"line-numbers\">\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L1\" rel=\"file-logistic_isolate_and_plot-py-L1\">1<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L2\" rel=\"file-logistic_isolate_and_plot-py-L2\">2<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L3\" rel=\"file-logistic_isolate_and_plot-py-L3\">3<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L4\" rel=\"file-logistic_isolate_and_plot-py-L4\">4<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L5\" rel=\"file-logistic_isolate_and_plot-py-L5\">5<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L6\" rel=\"file-logistic_isolate_and_plot-py-L6\">6<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L7\" rel=\"file-logistic_isolate_and_plot-py-L7\">7<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L8\" rel=\"file-logistic_isolate_and_plot-py-L8\">8<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L9\" rel=\"file-logistic_isolate_and_plot-py-L9\">9<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L10\" rel=\"file-logistic_isolate_and_plot-py-L10\">10<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L11\" rel=\"file-logistic_isolate_and_plot-py-L11\">11<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L12\" rel=\"file-logistic_isolate_and_plot-py-L12\">12<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L13\" rel=\"file-logistic_isolate_and_plot-py-L13\">13<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L14\" rel=\"file-logistic_isolate_and_plot-py-L14\">14<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L15\" rel=\"file-logistic_isolate_and_plot-py-L15\">15<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L16\" rel=\"file-logistic_isolate_and_plot-py-L16\">16<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L17\" rel=\"file-logistic_isolate_and_plot-py-L17\">17<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L18\" rel=\"file-logistic_isolate_and_plot-py-L18\">18<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L19\" rel=\"file-logistic_isolate_and_plot-py-L19\">19<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L20\" rel=\"file-logistic_isolate_and_plot-py-L20\">20<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L21\" rel=\"file-logistic_isolate_and_plot-py-L21\">21<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L22\" rel=\"file-logistic_isolate_and_plot-py-L22\">22<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L23\" rel=\"file-logistic_isolate_and_plot-py-L23\">23<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L24\" rel=\"file-logistic_isolate_and_plot-py-L24\">24<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L25\" rel=\"file-logistic_isolate_and_plot-py-L25\">25<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L26\" rel=\"file-logistic_isolate_and_plot-py-L26\">26<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L27\" rel=\"file-logistic_isolate_and_plot-py-L27\">27<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L28\" rel=\"file-logistic_isolate_and_plot-py-L28\">28<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L29\" rel=\"file-logistic_isolate_and_plot-py-L29\">29<\/span>\n            <span class=\"line-number\" id=\"file-logistic_isolate_and_plot-py-L30\" rel=\"file-logistic_isolate_and_plot-py-L30\">30<\/span>\n          <\/td>\n          <td class=\"line-data\">\n            <pre class=\"line-pre\"><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC1\"><span class=\"pl-st\">def<\/span> <span class=\"pl-en\">isolate_and_plot<\/span>(<span class=\"pl-vpf\">variable<\/span>):\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC2\">    <span class=\"pl-c\"># isolate gre and class rank<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC3\">    grouped <span class=\"pl-k\">=<\/span> pd.pivot_table(combos, <span class=\"pl-vpf\">values<\/span><span class=\"pl-k\">=<\/span>[<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>admit_pred<span class=\"pl-pds\">&#39;<\/span><\/span>], <span class=\"pl-vpf\">rows<\/span><span class=\"pl-k\">=<\/span>[variable, <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>prestige<span class=\"pl-pds\">&#39;<\/span><\/span>],\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC4\">                             <span class=\"pl-vpf\">aggfunc<\/span><span class=\"pl-k\">=<\/span>np.mean)\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC5\">    \n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC6\">    <span class=\"pl-c\"># in case you&#39;re curious as to what this looks like<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC7\">    <span class=\"pl-c\"># print grouped.head()<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC8\">    <span class=\"pl-c\">#                      admit_pred<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC9\">    <span class=\"pl-c\"># gre        prestige            <\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC10\">    <span class=\"pl-c\"># 220.000000 1           0.282462<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC11\">    <span class=\"pl-c\">#            2           0.169987<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC12\">    <span class=\"pl-c\">#            3           0.096544<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC13\">    <span class=\"pl-c\">#            4           0.079859<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC14\">    <span class=\"pl-c\"># 284.444444 1           0.311718<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC15\">    \n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC16\">    <span class=\"pl-c\"># make a plot<\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC17\">    colors <span class=\"pl-k\">=<\/span> <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>rbgyrbgy<span class=\"pl-pds\">&#39;<\/span><\/span>\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC18\">    <span class=\"pl-k\">for<\/span> col <span class=\"pl-k\">in<\/span> combos.prestige.unique():\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC19\">        plt_data <span class=\"pl-k\">=<\/span> grouped.ix[grouped.index.get_level_values(<span class=\"pl-c1\">1<\/span>)<span class=\"pl-k\">==<\/span>col]\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC20\">        pl.plot(plt_data.index.get_level_values(<span class=\"pl-c1\">0<\/span>), plt_data[<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>admit_pred<span class=\"pl-pds\">&#39;<\/span><\/span>],\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC21\">                <span class=\"pl-vpf\">color<\/span><span class=\"pl-k\">=<\/span>colors[<span class=\"pl-s3\">int<\/span>(col)])\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC22\">&nbsp;\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC23\">    pl.xlabel(variable)\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC24\">    pl.ylabel(<span class=\"pl-s1\"><span class=\"pl-pds\">&quot;<\/span>P(admit=1)<span class=\"pl-pds\">&quot;<\/span><\/span>)\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC25\">    pl.legend([<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>1<span class=\"pl-pds\">&#39;<\/span><\/span>, <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>2<span class=\"pl-pds\">&#39;<\/span><\/span>, <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>3<span class=\"pl-pds\">&#39;<\/span><\/span>, <span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>4<span class=\"pl-pds\">&#39;<\/span><\/span>], <span class=\"pl-vpf\">loc<\/span><span class=\"pl-k\">=<\/span><span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>upper left<span class=\"pl-pds\">&#39;<\/span><\/span>, <span class=\"pl-vpf\">title<\/span><span class=\"pl-k\">=<\/span><span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>Prestige<span class=\"pl-pds\">&#39;<\/span><\/span>)\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC26\">    pl.title(<span class=\"pl-s1\"><span class=\"pl-pds\">&quot;<\/span>Prob(admit=1) isolating <span class=\"pl-pds\">&quot;<\/span><\/span> <span class=\"pl-k\">+<\/span> variable <span class=\"pl-k\">+<\/span> <span class=\"pl-s1\"><span class=\"pl-pds\">&quot;<\/span> and presitge<span class=\"pl-pds\">&quot;<\/span><\/span>)\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC27\">    pl.show()\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC28\">&nbsp;\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC29\">isolate_and_plot(<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>gre<span class=\"pl-pds\">&#39;<\/span><\/span>)\n<\/div><div class=\"line\" id=\"file-logistic_isolate_and_plot-py-LC30\">isolate_and_plot(<span class=\"pl-s1\"><span class=\"pl-pds\">&#39;<\/span>gpa<span class=\"pl-pds\">&#39;<\/span><\/span>)\n<\/div><\/pre>\n          <\/td>\n        <\/tr>\n      <\/table>\n    <\/div>\n\n          <\/div>\n          <div class=\"gist-meta\">\n            <a href=\"https://gist.github.com/glamp/5077306/raw/logistic_isolate_and_plot.py\" style=\"float:right\">view raw<\/a>\n            <a href=\"https://gist.github.com/glamp/5077306#file-logistic_isolate_and_plot-py\">logistic_isolate_and_plot.py<\/a>\n            hosted with &#10084; by <a href=\"https://github.com\">GitHub<\/a>\n          <\/div>\n        <\/div>\n<\/div>\n')

var max = 0;
var i = 999;
while (i >= 100) {
    var j = 999;
    while (j >= i) {
        var p = i * j;
        if (p <= max) {
            j = 0;
        } else {
            var t = p;
            var rev = 0;
            while (t != 0) {
                rev = rev * 10 + t % 10;
                t = t / 10;
            }
            if (rev == p) {
                max = p;
            }
        }
        j = j - 1;
    }
    i = i - 1;
}
printnum(max);
putc('\n');

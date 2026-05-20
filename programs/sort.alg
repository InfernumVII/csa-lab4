var n = 0;
var buf[64];
var c = getnum();
while (c != 0) {
    buf[n] = c;
    n = n + 1;
    c = getnum();
}
var i = 0;
while (i < n - 1) {
    var j = 0;
    while (j < n - 1 - i) {
        if (buf[j] > buf[j + 1]) {
            var tmp = buf[j];
            buf[j] = buf[j + 1];
            buf[j + 1] = tmp;
        }
        j = j + 1;
    }
    i = i + 1;
}
i = 0;
while (i < n) {
    printnum(buf[i]);
    putc(' ');
    i = i + 1;
}
putc('\n');

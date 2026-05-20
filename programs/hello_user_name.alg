print("What is your name?\n");
var buf[64];
var i = 0;
var c = getc();
while (c != 10) {
    buf[i] = c;
    i = i + 1;
    c = getc();
}
print("Hello, ");
var j = 0;
while (j < i) {
    putc(buf[j]);
    j = j + 1;
}
putc('!');
putc('\n');

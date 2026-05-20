var a[4];
var b[4];
a[0] = 10; a[1] = 20; a[2] = 30; a[3] = 40;
b[0] = 1; b[1] = 2; b[2] = 3; b[3] = 4;
var dot = 0;
var i = 0;
while (i < 4) {
    dot = dot + a[i] * b[i];
    i = i + 1;
}
printnum(dot);
putc('\n');

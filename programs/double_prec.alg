var lo1 = 4294967295;
var hi1 = 0;
var lo2 = 1;
var hi2 = 0;
var rlo = lo1 + lo2;
var c = carry();
var rhi = hi1 + hi2;
if (c) {
    rhi = rhi + 1;
}
print("hi=");
printu(rhi);
print(" lo=");
printu(rlo);
putc('\n');

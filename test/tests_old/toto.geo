// This code was created by pygmsh v4.3.6.
p96 = newp;
Point(p96) = {-1, -1, -1, 1};
p97 = newp;
Point(p97) = {1, -1, -1, 1};
p98 = newp;
Point(p98) = {1, 1, -1, 1};
p99 = newp;
Point(p99) = {-1, 1, -1, 1};
l96 = newl;
Line(l96) = {p96, p97};
l97 = newl;
Line(l97) = {p97, p98};
l98 = newl;
Line(l98) = {p98, p99};
l99 = newl;
Line(l99) = {p99, p96};
ll24 = newll;
Line Loop(ll24) = {l96, l97, l98, l99};
s24 = news;
Plane Surface(s24) = {ll24};
Physical Surface(100) = {s24};
ex1[] = Extrude {0,0,2} {Surface{s24};};
Physical Surface(105) = {ex1[0]};
Physical Surface(101) = {ex1[2]};
Physical Surface(102) = {ex1[3]};
Physical Surface(103) = {ex1[4]};
Physical Surface(104) = {ex1[5]};
Physical Volume(10) = {ex1[1]};
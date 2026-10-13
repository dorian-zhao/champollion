-- Deterministic SQLite workload; output is compared with the native build's (basics.expected).
.headers on
.mode list
CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, score REAL, grp INTEGER);
WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 5000)
INSERT INTO t SELECT i, 'name' || (i * 7 % 101), (i * 37 % 1000) / 7.0, i % 13 FROM n;
CREATE INDEX t_grp ON t(grp, score);
SELECT count(*), sum(score), avg(score), min(name), max(name) FROM t;
SELECT grp, count(*), printf('%.6f', avg(score)) FROM t GROUP BY grp ORDER BY grp LIMIT 5;
SELECT id, score, rank() OVER (PARTITION BY grp ORDER BY score DESC, id) AS r
  FROM t WHERE grp = 3 ORDER BY r LIMIT 5;
SELECT a.id, b.id FROM t a JOIN t b ON a.score = b.score AND a.id < b.id ORDER BY a.id, b.id LIMIT 5;
SELECT json_object('n', 1, 'list', json_array(1, 2.5, 'x')), json_extract('{"a":{"b":[10,20]}}', '$.a.b[1]');
SELECT printf('%.17g', 0.1 + 0.2), 1e308 * 10, round(2.5), round(-2.5), 7 / 2, 7 % 3, cast('12abc' AS INTEGER);
SELECT upper('héllo'), length('héllo'), substr('abcdef', 2, 3), replace('aaa', 'a', 'bb'), hex('AZ');
BEGIN; DELETE FROM t WHERE grp > 5; ROLLBACK;
SELECT count(*) FROM t;
EXPLAIN QUERY PLAN SELECT * FROM t WHERE grp = 2 AND score > 10;

// Exercises the Go runtime on riscv64 Linux under rvrun: goroutines on many threads,
// channels, select with timers, maps, the garbage collector (including preempting a busy
// goroutine), files, and a TCP connection over loopback (netpoller on epoll). Output is
// deterministic, so the native build defines the expected output.
package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func goroutines() {
	var wg sync.WaitGroup
	var total atomic.Int64
	results := make(chan int, 100)
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sum := 0
			for j := 0; j <= i*1000; j++ {
				sum += j
			}
			total.Add(int64(sum))
			results <- i
		}(i)
	}
	wg.Wait()
	close(results)
	n := 0
	for range results {
		n++
	}
	fmt.Println("goroutines:", n, total.Load())
}

func selectAndTimers() {
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	done := time.After(30 * time.Millisecond)
	ticks := 0
	for {
		select {
		case <-tick.C:
			ticks++
		case <-done:
			fmt.Println("timers: ticked", ticks > 5)
			return
		}
	}
}

var sink uint64

func gc() {
	stop := make(chan struct{})
	go func() {
		var x uint64
		for {
			select {
			case <-stop:
				sink = x
				return
			default:
			}
			for i := 0; i < 1e7; i++ {
				x += uint64(i)
			}
		}
	}()
	m := map[string][]int{}
	for i := 0; i < 50000; i++ {
		k := fmt.Sprintf("key%d", i%1000)
		m[k] = append(m[k], i)
	}
	runtime.GC()
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	close(stop)
	fmt.Println("gc: maps", len(keys), keys[0], keys[len(keys)-1], len(m["key7"]))
}

func files() {
	dir, err := os.MkdirTemp("", "go-tests")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(path, []byte(strings.Repeat("champollion\n", 100)), 0o644); err != nil {
		panic(err)
	}
	f, _ := os.Open(path)
	defer f.Close()
	lines := 0
	s := bufio.NewScanner(f)
	for s.Scan() {
		lines++
	}
	entries, _ := os.ReadDir(dir)
	fmt.Println("files:", lines, len(entries), entries[0].Name())
}

func tcp() {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			panic(err)
		}
		defer c.Close()
		line, _ := bufio.NewReader(c).ReadString('\n')
		fmt.Fprintf(c, "echo: %s", strings.ToUpper(line))
	}()
	c, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		panic(err)
	}
	defer c.Close()
	fmt.Fprintf(c, "hello over tcp\n")
	reply, _ := bufio.NewReader(c).ReadString('\n')
	fmt.Print("tcp: ", reply)
}

func main() {
	goroutines()
	selectAndTimers()
	gc()
	files()
	tcp()
}

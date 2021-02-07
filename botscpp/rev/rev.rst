
/home/setepenre/.steam/ubuntu12_32/steam-runtime/run.sh
/media/setepenre/local/SteamLibraryLinux/steamapps/common/dota2/game/bin/linuxsteamrt64/dota2


.. code-block:: bash

    source botscpp/utilities/steam_setup.sh
    lldb-8 dota2/game/bin/linuxsteamrt64/dota2

    break set -r '^Init$' -s botcpp_radiant.so
    break set -r '^Observe$' -s botcpp_radiant.so
    break set -r '^Act$' -s botcpp_radiant.so

    process launch -- -dedicated +dota_1v1_skip_strategy 1 -gl -botworldstatesocket_threaded -botworldstatetosocket_frames 4 -botworldstatetosocket_radiant 12120 -botworldstatetosocket_dire 12121 -con_logfile console.log -con_timestamp -console -dev -insecure -noip -nowatchdog +clientport 27006 +dota_surrender_on_disconnect 0 +host_timescale 2 +hostname dotaservice +sv_cheats 1 +sv_hibernate_when_empty 0 -fill_with_bots +map start gamemode 21


.. code-block:: bash

    About to load botcpp from /media/setepenre/local/SteamLibraryLinux/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots/botcpp_radiant.so
    1 location added to breakpoint 1
    1 location added to breakpoint 2
    1 location added to breakpoint 3
    Process 23810 stopped
    * thread #1, name = 'dota2', stop reason = breakpoint 1.1
        frame #0: 0x00007fffe5feb8da botcpp_radiant.so`Init
    botcpp_radiant.so`Init:
    ->  0x7fffe5feb8da <+0>: pushq  %rbp
        0x7fffe5feb8db <+1>: movq   %rsp, %rbp
        0x7fffe5feb8de <+4>: subq   $0x20, %rsp
        0x7fffe5feb8e2 <+8>: movl   %edi, -0x4(%rbp)
    (lldb) bt
    frame #0: 0x00007fffe5feb8da botcpp_radiant.so`Init
    frame #1: 0x00007fffebab9d9f libserver.so
    frame #2: 0x00007fffebafceb6 libserver.so
    frame #3: 0x00007fffeba71d86 libserver.so
    frame #4: 0x00007fffeb3807ce libserver.so
    frame #5: 0x00007fffeb396849 libserver.so
    frame #6: 0x00007fffeb47de62 libserver.so
    frame #7: 0x00007ffff6f8000b libengine2.so
    frame #8: 0x00007ffff6f78b39 libengine2.so
    frame #9: 0x00007ffff6f79245 libengine2.so
    frame #10: 0x00007ffff6f6b8e3 libengine2.so
    frame #11: 0x00007ffff6f71841 libengine2.so
    frame #12: 0x00007ffff6f71b63 libengine2.so`Source2Main + 211
    frame #13: 0x00005555555550a0 dota2
    frame #14: 0x0000555555554eae dota2
    frame #15: 0x00007ffff797c0b3 libc.so.6`__libc_start_main(main=0x0000555555554d10, argc=37, argv=0x00007fffffffda38, init=<unavailable>, fini=<unavailable>, rtld_fini=<unavailable>, stack_end=0x00007fffffffda28) at libc-start.c:308:16
    (lldb) thread list
    Process 23810 stopped
    * thread #1: tid = 23810, 0x00007fffe5feb8da botcpp_radiant.so`Init, name = 'dota2', stop reason = breakpoint 1.1
      thread #2: tid = 24009, 0x00007ffff7b577b1 libpthread.so.0`__pthread_cond_timedwait at futex-internal.h:320:13, name = 'GlobPool/0'
      thread #3: tid = 24010, 0x00007ffff7b577b1 libpthread.so.0`__pthread_cond_timedwait at futex-internal.h:320:13, name = 'dota2'
      thread #4: tid = 24011, 0x00007ffff7b577b1 libpthread.so.0`__pthread_cond_timedwait at futex-internal.h:320:13, name = 'AsyncIOService'
      thread #5: tid = 24012, 0x00007ffff7b577b1 libpthread.so.0`__pthread_cond_timedwait at futex-internal.h:320:13, name = 'OnAsyncProcessR'
      thread #6: tid = 24013, 0x00007ffff7b577b1 libpthread.so.0`__pthread_cond_timedwait at futex-internal.h:320:13, name = 'AsyncTextureHoo'
      thread #7: tid = 24014, 0x00007ffff7a6aaff libc.so.6`__GI___poll(fds=0x000055555e2a6a00, nfds=1, timeout=-1) at poll.c:29:10, name = 'IOCP Thread 0'
      thread #8: tid = 24021, 0x00007ffff7a6aaff libc.so.6`__GI___poll(fds=0x00007fffe6e33a08, nfds=1, timeout=5000) at poll.c:29:10, name = 'CJobMgr::m_Work'
      thread #9: tid = 24015, 0x00007ffff7a775ce libc.so.6`epoll_wait(epfd=109, events=0x00007fffe7239c10, maxevents=1, timeout=49) at epoll_wait.c:30:10, name = 'CIPCServer::Thr'
      thread #10: tid = 24017, 0x00007ffff7b57376 libpthread.so.0`__pthread_cond_wait at futex-internal.h:183:13, name = 'CFileWriterThre'
      thread #11: tid = 24018, 0x00007ffff7a6d3eb libc.so.6`fdatasync(fd=126) at fdatasync.c:28:10, name = 'CApplicationMan'
      thread #13: tid = 24104, 0x00007ffff7a6aaff libc.so.6`__GI___poll(fds=0x00007fffe6c018b8, nfds=1, timeout=5000) at poll.c:29:10, name = 'CHTTPClientThre'
      thread #14: tid = 24105, 0x00007ffff7a6aaff libc.so.6`__GI___poll(fds=0x00007fffe6b023e0, nfds=2, timeout=5000) at poll.c:29:10, name = 'dota2'
    (lldb) image list
    [  0] 547B0BC1-EDF2-C8BE-F4D8-D832645C4712-52A8BA1C 0x0000555555554000 /media/setepenre/local/SteamLibraryLinux/steamapps/common/dota2/game/bin/linuxsteamrt64/dota2
    ...
    [ 47] 450BA72B-7B50-AD71-9649-88D09F0AC916-27245CCE 0x00007fffe5f28000 /media/setepenre/local/SteamLibraryLinux/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots/botcpp_radiant.so







image dump symtab libserver.so

watch set var global

-dedicated +dota_1v1_skip_strategy 1 -gl -botworldstatesocket_threaded -botworldstatetosocket_frames 4 -botworldstatetosocket_radiant 12120 -botworldstatetosocket_dire 12121 -con_logfile console.log -con_timestamp -console -dev -insecure -noip -nowatchdog +clientport 27006 +dota_surrender_on_disconnect 0 +host_timescale 2 +hostname dotaservice +sv_cheats 1 +sv_hibernate_when_empty 0 -fill_with_bots +map start gamemode 21

disassemble -a 0x1234


(lldb) break set -r . -s libobjc.A.dylib


The -s option takes a shared library as its value, and that limits the breakpoint to the specified shared library. You can specify the -s option more than once to specify more than one shared library for inclusion in the breakpoint search.

The -r option's value is a regular expression; if the symbol name matches that expression, it will be included in the breakpoint. . matches everything.

The lldb tutorial:

http://lldb.llvm.org/tutorial.html

starts with a description of the structure of lldb commands that you might find helpful.

from https://stackoverflow.com/questions/44928511/gdb-lldb-break-at-all-functions-of-specified-module-shared-library



(lldb) image dump symtab libserver.so
Symtab, file = /media/setepenre/local/SteamLibraryLinux/steamapps/common/dota 2 beta/game/dota/bin/linuxsteamrt64/libserver.so, num_symbols = 11250:
               Debug symbol
               |Synthetic symbol
               ||Externally Visible
               |||



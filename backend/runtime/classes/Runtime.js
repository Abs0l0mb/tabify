'use strict';

import { RuntimeData } from './RuntimeData.js';
import { Log } from './Log.js';
import { Tools } from './Tools.js';
import { fork } from 'child_process';
import * as readline from 'readline';
import * as nodePath from 'path';
import * as chokidar from '../lib/chokidar.cjs';

export class Runtime {

    constructor(srcDir, mode, typeChecking) {

        this.srcDir = srcDir;
        this.mode = mode;
        this.typeChecking = typeChecking;
        this.processes = {};
        this.watcher = null;
        this.refreshing = false;
        this.lastWatch = {
            path: null,
            event: null,
            time: Date.now()
        };

        this.init();
    }

    /*
    **
    **
    */
    async init() {

        if (this.mode === "dev") {
            this.watch();
            this.listenForReadlineRequest();
        }

        const entryPoint = await this.getBuiltEntryPoint();

        if (!entryPoint)
            return;

        if (this.mode === "dev")
            this.devRun(entryPoint);

        else if (this.mode === "build") {
            Log.log(`${Log.Dim}${Log.Bright}Built entry point: ${Log.Reset}${Log.FgGreen}${Log.Bright}${entryPoint}${Log.Reset}`);
            await this.killChilds();
            process.exit();
        }
    }

    /*
    **
    **
    */
    listenForReadlineRequest() {

        readline.emitKeypressEvents(process.stdin);
        
        if (process.stdin.isTTY)
            process.stdin.setRawMode(true);

        process.stdin.on("keypress", async (key, data) => {

            if (data.ctrl && data.name === 't') {
                this.typeChecking = !this.typeChecking;
                await this.devRefresh();
            }

            if (data.ctrl && ['b', 'r'].includes(data.name))
                await this.devRefresh();

            if (data.ctrl && ['w'].includes(data.name))
                this.toggleWatching();
        });
    }

    /*
    **
    **
    */
    async getBuiltEntryPoint() {

        try {

            Log.cyan(`Type checking ${this.typeChecking ? 'enabled' : 'disabled'}`);
            Log.cyan(`Watching ${this.watcher ? 'enabled' : 'disabled'}`);
            
            const startTime = Date.now();

            if (this.mode === 'dev')
                Log.startLoading();

            const entryPoint = await new Promise((resolve, reject) => {
    
                const builderProcess = fork(nodePath.resolve(process.cwd(), 'runtime/build-process', 'index.js'), [
                    this.mode,
                    this.typeChecking ? "on" : "off",
                    this.srcDir
                ]);

                this.processes[builderProcess.pid] = builderProcess;
                
                builderProcess.on('message', (json) => {

                    const message = JSON.parse(json);

                    if (message.type === 'success')
                        resolve(message.entryPoint);
                    
                    else if (message.type === 'stopped-because-type-checking-error')
                        reject('type-checking-error');

                    else if (message.type === 'stopped')
                        reject('build-failed');
                });

                builderProcess.on('close', () => {
                    delete(this.processes[builderProcess.pid]);
                });
            });

            if (this.mode === 'dev')
                Log.stopLoading();

            Log.blue(`Build done (${Tools.humanSince(Date.now() - startTime)})`);

            return entryPoint;

        } catch(error) {

            if (this.mode === 'dev')
                Log.stopLoading();

            if (error === 'type-checking-error')
                Log.red('Build failed due to type checking error');

            else if (error === 'build-failed')
                Log.red('Build failed');

            else {
                    
                Log.red('Error when instanciating build process');
                Log.printError(error);

                if (this.mode === 'build') {
                    await this.killChilds();
                    process.exit();
                }
            }

            return null;
        }
    }

    /*
    **
    **
    */
    devRun(entryPoint) {

        if (!entryPoint)
            return;

        const execArgv = [];
        
        if (this.mode === 'dev')
            execArgv.push('--enable-source-maps');

        const runProcess = fork(entryPoint, [RuntimeData.getOptions().mainProcess.devPort], {
            execArgv: execArgv,
            silent: true
        });

        this.processes[runProcess.pid] = runProcess;

        runProcess.on('close', () => {
            delete(this.processes[runProcess.pid]);
        });

        runProcess.stdout.on('data', (data) => {
            Log.log(data.toString(), false);
        });

        runProcess.stderr.on('data', (data) => {
            Log.red(data.toString(), false);
        });
    }

    /*
    **
    **
    */
    watch() {

        if (this.watcher)
            return;

        this.watcher = chokidar.watch(this.srcDir, {
            ignoreInitial: true
        }).on('all', this.onFSWatch.bind(this));
    }

    /*
    **
    **
    */
    onFSWatch(event, path) {

        if (Date.now() - this.lastWatch.time > 150 || event !== this.lastWatch.event || path !== this.lastWatch.path) {

            this.devRefresh();
            
            this.lastWatch = {
                event: event,
                path: path,
                time: Date.now()
            };
        }
    }

    /*
    **
    **
    */
    async devRefresh() {
        
        if (this.refreshing)
            return Log.yellow(`Refresh already requested`);;
        
        this.refreshing = true;

        Log.blue(`Refresh requested...`);

        await this.killChilds();

        this.refreshing = false;

        Log.clear();
        
        Log.blue(`Refreshed on ${new Date().toLocaleString('en')}`);
        Log.blue(`Up time: ${Tools.humanSince(Date.now() - process.env.START_TIME)}`);

        this.devRun(await this.getBuiltEntryPoint());
    }

    /*
    **
    **
    */
    toggleWatching() {

        if (this.watcher) {
            this.watcher.close();
            this.watcher = null;
            Log.cyan(`Watching disabled`);
        }
        else {
            this.watch();
            Log.cyan(`Watching enabled`);
        }
    }

    /*
    **
    **
    */
    killChilds() {

        return new Promise(resolve => {
            
            let expired = false;

            const closePromises = [];

            for (const pid in this.processes) {
                this.processes[pid].kill('SIGTERM');
                closePromises.push(new Promise(closed => {
                    this.processes[pid].once('close', () => {
                        delete this.processes[pid];
                        closed();
                    });
                }));
            }

            (async () => {
                await Promise.all(closePromises);
                if (expired)
                    return;
                expired = true;
                resolve();
            })();

            (async () => {
                await Tools.sleep(1000);
                if (expired)
                    return;
                expired = true;
                for (const pid in this.processes)
                    this.processes[pid].kill('SIGKILL');
                await Tools.sleep(50);
                for (const pid in this.processes)
                    this.processes[pid].kill('SIGKILL');
                this.processes = {};
                resolve();
            })();
        });
    }
}
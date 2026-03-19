'use strict';

import { RuntimeData } from './RuntimeData.js';
import { Log } from './Log.js';
import { Tools } from './Tools.js';
import { DevToolServer } from './dev-tool-server/DevToolServer.js';
import { HttpStaticServer } from './http-static-server/HttpStaticServer.js';

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

        this.devServer = null;
        this.devToolServer = null;

        this.status = 'building';
        this.bundlingSystem = 'sourcemap';

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
            await this.devServe();
        }

        const distDir = await this.build();

        if (this.mode === "build") {
            Log.log(`${Log.Dim}${Log.Bright}Distribution directory: ${Log.Reset}${Log.FgGreen}${Log.Bright}${distDir}${Log.Reset}`);
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

            if (data.ctrl && data.name === 's') {
                this.toggleBundlingSystem();
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
    async devServe() {

        if (this.devServer) {
            
            await this.devServer.close();

            Log.blue(`Http server closed`);
        }

        await new Promise(function(resolve) {

            this.devServer = new HttpStaticServer();

            this.devToolServer = new DevToolServer(this);

            const options = RuntimeData.getOptions();

            if (options.apiPort)
                this.devServer.setProxy(`http://localhost:${options.apiPort}`);

            this.devServer.listen({
                port: options.devPort, 
                host: '0.0.0.0'
            }, function() {

                Log.blue(`Http server listening on ${options.devPort}`);

                resolve();
            
            }.bind(this));

        }.bind(this));
    }

    /*
    **
    **
    */
    async build() {

        try {

            Log.cyan(`Type checking ${this.typeChecking ? 'enabled' : 'disabled'}`);

            if (this.mode === 'dev') {
                Log.cyan(`Bundling system set to ${this.bundlingSystem}`);
                Log.cyan(`Watching ${this.watcher ? 'enabled' : 'disabled'}`);
            }

            this.setStatus('building');

            const startTime = Date.now();

            if (this.mode === 'dev')
                Log.startLoading();

            this.reloadStatics();

            const distDir = await new Promise((resolve, reject) => {
    
                const builderProcess = fork(nodePath.resolve(process.cwd(), 'runtime/build-process', 'index.js'), [
                    this.mode,
                    this.typeChecking ? "on" : "off",
                    this.bundlingSystem,
                    this.srcDir
                ]);

                this.processes[builderProcess.pid] = builderProcess;
                
                builderProcess.on('message', (json) => {

                    const message = JSON.parse(json);

                    if (message.type === 'success')
                        resolve(message.distDir);
                    
                    else if (message.type === 'stopped-because-type-checking-error')
                        reject('type-checking-error');

                    else if (message.type === 'stopped')
                        reject('build-failed');
                });

                builderProcess.on('close', () => {
                    delete(this.processes[builderProcess.pid]);
                });
            });

            this.reloadStatics(distDir);

            this.setStatus('done');

            if (this.devToolServer)
                this.devToolServer.broadcast('reload');

            if (this.mode === 'dev')
                Log.stopLoading();

            Log.blue(`Build done (${Tools.humanSince(Date.now() - startTime)})`);

            return distDir;

        } catch(error) {

            if (this.mode === 'dev')
                Log.stopLoading();

            this.setStatus('error');

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

            return false;
        }
    }

    /*
    **
    **
    */
    reloadStatics(distDir) {

        if (!this.devServer)
            return;

        this.devServer.clearStatics();
        this.devServer.addStatics(distDir);

        const options = RuntimeData.getOptions();

        const pages = options.pages;

        for (let key in pages)
            pages[key].title = pages[key].title ? `${options.appName} - ${pages[key].title}` : options.appName;

        this.devServer.clearPages();
        this.devServer.addPages(pages, distDir);
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
        
        Log.blue(`Refreshed on ${new Date().toLocaleString('fr')}`);
        Log.blue(`Up time: ${Tools.humanSince(Date.now() - process.env.START_TIME)}`);

        await this.build();
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
    toggleTypeChecking() {

        this.typeCheckingStatus = this.typeCheckingStatus === 'on' ? 'off' : 'on';

        if (this.devToolServer)
            this.devToolServer.broadcastTypeCheckingStatus();
    }

    /*
    **
    **
    */
    toggleBundlingSystem() {

        if (this.bundlingSystem === 'fast')
            this.bundlingSystem = 'sourcemap';
        else if (this.bundlingSystem === 'sourcemap')
            this.bundlingSystem = 'production';
        else if (this.bundlingSystem === 'production')
            this.bundlingSystem = 'fast';

        if (this.devToolServer)
            this.devToolServer.broadcastBundlingSystem();
    }

    /*
    **
    **
    */
    setStatus(status) {

        this.status = status;

        if (this.devToolServer)
            this.devToolServer.broadcastBuilderStatus();
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
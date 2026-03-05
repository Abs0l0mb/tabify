'use strict';

process.env.LOG_ID = 'runtime';
process.env.START_TIME = Date.now();
process.env.TIME_ZONE = 'Europe/Paris';

import { Runtime } from './classes/Runtime.js';
import { RuntimeData } from './classes/RuntimeData.js';
import { Log } from './classes/Log.js';
import * as nodePath from 'path';
import * as readline from 'readline';

class RuntimeHandler {

    constructor() {
        
        this.srcDir = nodePath.resolve(process.cwd(), 'src');
        
        this.mode = null;
        this.typeChecking = false;
        this.readlineInterface = null;
        
        this.initMode();
        this.initTypeChecking();
        this.initLogs();
        this.initTerminationListeners();

        this.runtime = new Runtime(this.srcDir, this.mode, this.typeChecking);
    }

    /*
    **
    **
    */
    initMode() {

        if (process.argv.includes('-d') || process.argv.includes('--dev'))
            this.mode = 'dev';
        else if (process.argv.includes('-b') || process.argv.includes('--build'))
            this.mode = 'build';
        else
            throw new Error('Mode argument is incorrect or unspecified, expected: -d (--dev) or -b (--build)');
    }

    /*
    **
    **
    */
    initTypeChecking() {

        if (process.argv.includes('-t') || process.argv.includes('--type-checking'))
            this.typeChecking = true;
    }

    /*
    **
    **
    */
    initLogs() {

        if (this.mode === 'dev');
            Log.clear();

        this.displayStartupLogs();
        
        Log.clearCallback = this.displayStartupLogs.bind(this);
        
        Log.blue(`Started on ${new Date().toLocaleString('fr')}`);
    }

    /*
    **
    **
    */
    displayStartupLogs() {

        Log.blank();
        Log.green(`  ${RuntimeData.getOptions().runtimeName} ${Log.Dim}@${this.mode}${Log.Reset}`, false);
        Log.blank();
    }

    /*
    **
    **
    */
    initTerminationListeners() {

        this.readlineInterface = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        for (const signal of [
            'SIGINT',
            'SIGQUIT',
            'SIGTERM',
            'SIGKILL'
        ]) {
            
            this.readlineInterface.on(signal, () => {
                this.kill(signal);
            });
        }
    }

    /*
    **
    **
    */
    async kill(signal) {
        
        Log.blue(`${signal} received, exiting...`);

        Log.stopLoading();

        this.readlineInterface.close();

        if (this.runtime)
            await this.runtime.killChilds();

        process.exit();
    }
}

try {
    new RuntimeHandler();
}
catch(error) {
    Log.printError(error);
}
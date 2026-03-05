'use strict';

process.env.LOG_ID = 'build';

import { Build } from '../classes/Build.js';
import { RuntimeData } from '../classes/RuntimeData.js';
import { Log } from '../classes/Log.js';

new class BuildHandler {
    
    constructor() {

        this.mode = process.argv[2];
        this.typeChecking = process.argv[3] === 'on';
        this.bundlingSystem = process.argv[4];
        this.srcDir = process.argv[5];

        this.build = null;
        
        this.listenForTermination();
        this.handleBuild();
    }
    
    /*
    **
    **
    */
    async handleBuild() {

        const options = RuntimeData.getOptions();

        this.build = new Build(
            this.mode,
            this.typeChecking,
            this.bundlingSystem,
            this.srcDir);
        
        try {

            await this.build.run();

            this.sendSuccess();

        } catch(error) {
                
            if (error.message === 'type-checking-error')
                this.sendStoppedDueToTypeCheckingError();

            else {
                
                Log.red('Build error');
                Log.printError(error);

                this.sendStopped();
            }

            process.exit();
        }
    }

    /*
    **
    **
    */
    sendStoppedDueToTypeCheckingError() {

        process.send(JSON.stringify({
            type: 'stopped-because-type-checking-error'
        }));
    }

    /*
    **
    **
    */
    sendStopped() {

        process.send(JSON.stringify({
            type: 'stopped'
        }));
    }

    /*
    **
    **
    */
    sendSuccess() {

        process.send(JSON.stringify({
            type: 'success',
            distDir: this.build.distModeDir
        }));
    }

    /*
    **
    **
    */
    listenForTermination() {
    
        process.once('SIGTERM', () => process.exit());
    }
}
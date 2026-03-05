'use strict';

process.env.LOG_ID = 'build';

import { Build } from '../classes/Build.js';
import { RuntimeData } from '../classes/RuntimeData.js';
import { Log } from '../classes/Log.js';
import * as fs from 'fs';

new class BuildHandler {
    
    constructor() {

        this.mode = process.argv[2];
        this.typeChecking = process.argv[3] === 'on';
        this.srcDir = process.argv[4];

        this.mainBuild = null;

        this.listenForTermination();
        this.handleBuilds();
    }
    
    /*
    **
    **
    */
    async handleBuilds() {

        const options = RuntimeData.getOptions();

        //============
        //MAIN PROCESS
        //============

        this.mainBuild = new Build(
            this.mode,
            this.typeChecking,
            options.mainProcess.distName,
            options.mainProcess.entryPoint,
            this.srcDir,
            true);
        
        await this.handleBuild(this.mainBuild);

        //=============
        //SUB PROCESSES
        //=============

        if (!Array.isArray(options.subProcesses))
            return;

        for (const subProcess of options.subProcesses) {

            const subProcessBuild = new Build(
                this.mode,
                this.typeChecking,
                subProcess.distName,
                subProcess.entryPoint,
                this.srcDir,
                false);

            await this.handleBuild(subProcessBuild);

            if (subProcess.instanciationKey) {

                let content = fs.readFileSync(this.mainBuild.outputEntryPoint);
                content = content.toString().replace(new RegExp(subProcess.instanciationKey, 'g'), subProcessBuild.outputEntryPoint);
                fs.writeFileSync(this.mainBuild.outputEntryPoint, content);
            }
        }
        
        return this.sendSuccess();
    }

    /*
    **
    **
    */
    async handleBuild(buildInstance) {
    
        try {

            await buildInstance.run();

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
            entryPoint: this.mainBuild.outputEntryPoint
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
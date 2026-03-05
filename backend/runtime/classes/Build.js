'use strict';

import { RuntimeData } from './RuntimeData.js';
import { Tools } from './Tools.js';
import { Log } from './Log.js';

import ts from 'typescript';
import * as fs from 'fs';
import * as nodePath from 'path';
import * as esbuild from 'esbuild';

export class Build {

    constructor(mode, typeChecking, distName, entryPoint, srcDir, isMain) {

        this.mode = mode;
        this.typeChecking = typeChecking;
        this.distName = distName;
        this.entryPoint = entryPoint;
        this.srcDir = srcDir;
        this.isMain = isMain;

        this.distDir = nodePath.resolve(process.cwd(), `runtime/dist`);
        this.distProjectDir = nodePath.resolve(this.distDir, this.distName);
        this.distProjectModeDir = nodePath.resolve(this.distProjectDir, this.mode);

        this.entryPoint = nodePath.resolve(this.srcDir, this.entryPoint);
        this.outputEntryPoint = nodePath.resolve(this.distProjectModeDir, 'index.esb-min.js');

        this.storeDir = nodePath.resolve(process.cwd(), `store`);
    }

    /*
    **
    **
    */
    async run() {

        this.initDistDir();

        if (this.typeChecking)
            this.doTypeChecking();

        await this.runESBuild();

        if (this.mode === 'build')
            Tools.copyDirSync(this.storeDir, this.distProjectModeDir);
    }

    /*
    **
    **
    */
    initDistDir() {
        
        if (!fs.existsSync(this.distDir) || !fs.lstatSync(this.distDir).isDirectory())
            fs.mkdirSync(this.distDir);

        if (!fs.existsSync(this.distProjectDir) || !fs.lstatSync(this.distProjectDir).isDirectory())
            fs.mkdirSync(this.distProjectDir);

        Tools.rmDirSync(this.distProjectModeDir);
        fs.mkdirSync(this.distProjectModeDir);
    }

    /*
    **
    **
    */
    doTypeChecking() {
        
        Log.blue(`Type checking ${this.distName}...`);

        const program = ts.createProgram([this.entryPoint], {
            outDir: this.distProjectModeDir,
            target: ts.ScriptTarget.ES2015,
            module: ts.ModuleKind.CommonJS,
            esModuleInterop: true,
            strictNullChecks: true,
            strictPropertyInitialization: false,
            strictFunctionTypes: true,
            noImplicitReturns: true,
            noImplicitThis: false,
            noImplicitAny: false,
            skipLibCheck: true,
            baseUrl: '.',
            paths: {
                '@src/*': ['src/*']
            }
        });
        
        program.emit();
        
        const diagnostics = ts.getPreEmitDiagnostics(program);
        
        const files = [];

        for (const diagnostic of diagnostics) {

            if (diagnostic.file) {

                const { line, character } = ts.getLineAndCharacterOfPosition(diagnostic.file, diagnostic.start);
                const message = ts.flattenDiagnosticMessageText(diagnostic.messageText);
                const parsedPath = nodePath.parse(diagnostic.file.fileName);

                files.push({
                    dir: parsedPath.dir,
                    name: parsedPath.base,
                    line: line,
                    position: character,
                    message: message
                });

            } else {
                
                Log.yellow(`! ${ts.flattenDiagnosticMessageText(diagnostic.messageText)}`);
            }
        }

        files.sort(function(a, b) {
            if ( a.dir.length < b.dir.length )
                return -1;
            if ( a.dir.length > b.dir.length )
                return 1;
            return 0;
        });
          
        if (files.length > 0) {

            Log.red(`[${this.distName}] TypeScript compilation failed throwing following error(s):`);

            for (const file of files) {

                Log.log(`${Log.FgRed}${Log.Bright}! ${Log.Reset}`
                    + `${Log.FgBlue}${file.dir}/${Log.Bright}${Log.FgCyan} ${file.name}${Log.Reset}`
                    + `${Log.FgGreen}${Log.Bright} ${file.line + 1} ${Log.Reset}`
                    + `${file.message}`);
            }

            Log.red(`${files.length} error(s) found`);

            throw new Error('type-checking-error');
        }
    }

    /*
    **
    **
    */
    async runESBuild() {

        Log.blue(`Building ${this.distName} for ${this.mode === 'build' ? 'production' : 'development'}...`);

        const dependencies = Object.keys(RuntimeData.getPackage().dependencies);
        
        const defined = {
            '__MODE__': JSON.stringify(this.mode === 'build' ? 'production' : 'development')
        };

        await esbuild.build({
            entryPoints: [this.entryPoint],
            outfile: this.outputEntryPoint,
            external: dependencies,
            platform: 'node',
            target: 'es2020',
            format: 'esm',
            bundle: true,
            minify: true,
            sourcemap: this.mode === 'dev' ? 'inline' : undefined,
            legalComments: 'none',
            define: defined
        });
    }
}
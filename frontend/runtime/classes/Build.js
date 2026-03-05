'use strict';

import { RuntimeData } from './RuntimeData.js';
import { Tools } from './Tools.js';
import { Log } from './Log.js';

import ts from 'typescript';
import * as fs from 'fs';
import * as nodePath from 'path';
import * as recast from 'recast';
import * as esbuild from 'esbuild';

import JavaScriptObfuscator from '../lib/javascript-obfuscator.cjs';
import Terser from '../lib/terser.cjs';
import sass from '../lib/sass.cjs';
import minimizeHTML from '../lib/minimize-html.cjs';

export class Build {

    constructor(mode, typeChecking, bundlingSystem, srcDir) {

        this.id = Tools.randStr(10);

        this.mode = mode;
        this.typeChecking = typeChecking;
        this.bundlingSystem = bundlingSystem;
        this.srcDir = srcDir;

        if (!['fast', 'sourcemap', 'production'].includes(this.bundlingSystem))
            this.bundlingSystem = 'fast';

        this.distDir = nodePath.resolve(process.cwd(), `runtime/dist`);
        this.distModeDir = nodePath.resolve(this.distDir, this.mode);

        this.mainTs = nodePath.resolve(this.srcDir, 'main.ts');
        this.jsTmp = nodePath.resolve(process.cwd(), 'runtime/tmp');
        this.jsDir = nodePath.resolve(this.distModeDir, 'js');
        this.jsName = `client-${this.id}.js`
        this.jsPath = nodePath.resolve(this.jsDir, this.jsName);
        this.jsLinks = [`/js/${this.jsName}`];

        this.mainScss = nodePath.resolve(this.srcDir, 'scss/main.scss');
        this.cssDir = nodePath.resolve(this.distModeDir, 'css');
        this.cssName = `styles-${this.id}.css`;
        this.cssPath = nodePath.resolve(this.cssDir, this.cssName);
        this.cssLinks = [`/css/${this.cssName}`];

        this.storeDir = nodePath.resolve(process.cwd(), `store`);

        this.runtimeOptions = RuntimeData.getOptions();
    }

    /*
    **
    **
    */
    async run() {
        
        this.initDirectories();
        this.copyAssetsDirectory();
        this.copyFavicon();
        
        if (this.typeChecking)
            this.doTypeChecking();

        Log.blue(`Building for ${this.mode === 'build' ? 'production' : 'development'}...`);

        await this.buildJS();
        await this.buildCSS();
        await this.buildHTML();
    }

    /*
    **
    **
    */
    initDirectories() {
        
        if (!fs.existsSync(this.distDir) || !fs.lstatSync(this.distDir).isDirectory())
            fs.mkdirSync(this.distDir);

        Tools.rmDirSync(this.distModeDir);
        fs.mkdirSync(this.distModeDir);

        if (!fs.existsSync(this.jsTmp) || !fs.lstatSync(this.jsTmp).isDirectory())
            fs.mkdirSync(this.jsTmp);

        if (!fs.existsSync(this.jsDir) || !fs.lstatSync(this.jsDir).isDirectory())
            fs.mkdirSync(this.jsDir);

        if (!fs.existsSync(this.cssDir) || !fs.lstatSync(this.cssDir).isDirectory())
            fs.mkdirSync(this.cssDir);
    }

    /*
    **
    **
    */
    copyAssetsDirectory() {

        Tools.copyDirSync(`${this.srcDir}/assets`, this.distModeDir);
    }

    /*
    **
    **
    */
    copyFavicon() {

        fs.copyFileSync(`${this.srcDir}/assets/favicon.ico`, `${this.distModeDir}/favicon.ico`);
    }

    /*
    **
    **
    */
    doTypeChecking() {
        
        Log.blue(`Type checking...`);

        const program = ts.createProgram([this.mainTs], {
            outDir: this.jsTmp,
            target: ts.ScriptTarget.ES2016,
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

            Log.red(`TypeScript compilation failed throwing following error(s):`);

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
    async buildJS() {

        //===================
        //VENDOR DEPENDENCIES
        //===================

        let dependencies = this.runtimeOptions.vendorDependencies;
        
        if (!Array.isArray(this.runtimeOptions.vendorDependencies))
            dependencies = [];

        const externals = [];
        const vendorParts = [];
        const vendorInstanciatorParts = [];

        for (let data of dependencies) {

            if (typeof data !== 'object' 
                || data === null
                || typeof data.import !== 'string' 
                || typeof data.from !== 'string')
                continue;

            externals.push(data.from);

            if (!['default', 'wildcard'].includes(data.importSyntax))
                data.importSyntax = 'default';
            
            if (data.importSyntax === 'default') {
                vendorParts.push(`
                    import ${data.import} from '${data.from}';
                    window['${data.import}'] = ${data.import};
                `);
            }
            else if (data.importSyntax === 'wildcard') {
                vendorParts.push(`
                    import * as ${data.import} from '${data.from}';
                    window['${data.import}'] = ${data.import};
                `);
            }

            vendorInstanciatorParts.push(`
                case '${data.from}': {
                    return window['${data.import}'];
                    break;
                }
            `); 
        }

        const vendor = vendorParts.join('');
        const vendorInstanciator = `
            window.require = (moduleName) => {
                switch (moduleName) {
                    ${vendorInstanciatorParts.join('')}
                }
            };
        `;

        const vendorPathIn = nodePath.resolve(this.jsTmp, 'BUILD__vendor.js');
        const vendorPathOut = nodePath.resolve(this.jsTmp, 'BUILD__vendor.esmin.js');
        const vendorInstanciatorPath = nodePath.resolve(this.jsTmp, 'BUILD__vendorInstanciator.js');

        fs.writeFileSync(vendorPathIn, vendor);
        fs.writeFileSync(vendorInstanciatorPath, vendorInstanciator);

        await esbuild.build({
            entryPoints: [vendorPathIn],
            outfile: vendorPathOut,
            platform: 'browser',
            target: 'es6',
            bundle: true,
            minify: true,
            sourcemap: false,
            legalComments: 'none'
        });

        //===
        //APP
        //===

        const defined = {
            __APP_NAME__: JSON.stringify(this.runtimeOptions.appName)
        };

        const storedFiles = fs.readdirSync(this.storeDir);
        
        for (const filename of storedFiles)
            defined[`__STORE_${filename.split('.json')[0]}__`] = this.packStoredData(filename);

        await esbuild.build({
            entryPoints: [this.mainTs],
            outfile: this.jsPath,
            platform: 'browser',
            target: 'es6',
            bundle: true,
            minify: true,
            sourcemap: this.mode === 'dev' && this.bundlingSystem === 'sourcemap' ? 'inline' : false,
            legalComments: 'none',
            external: externals,
            define: defined
        });

        //======
        //BUNDLE
        //======

        let appJs = fs.readFileSync(this.jsPath).toString();
        let vendorJs = fs.readFileSync(vendorPathOut).toString();
        
        const vendorInstanciatorJs = fs.readFileSync(vendorInstanciatorPath).toString();

        if (this.bundlingSystem === 'sourcemap') {

            vendorJs = this.runESTransform(vendorJs + vendorInstanciatorJs);

            const vendorJsName = `vendor-${this.id}.js`;
            const vendorJsPath = nodePath.resolve(this.jsDir, vendorJsName);

            fs.writeFileSync(vendorJsPath, vendorJs);
            
            this.jsLinks.unshift(`/js/${vendorJsName}`);
        }
        else {
            
            if (this.mode === 'build' || this.bundlingSystem === 'production') {

                Log.blue('Obfuscating...');

                appJs = this.transformObjectKeysToLitterals(appJs);
                appJs = this.runJSObfuscator(appJs);
                appJs = await this.runTerser(appJs);
            }

            appJs = this.runESTransform(vendorJs + vendorInstanciatorJs + appJs);
            
            fs.writeFileSync(this.jsPath, appJs);
        }
    }

    /*
    **
    **
    */
    packStoredData(storeFilename) {

        let json = fs.readFileSync(nodePath.resolve(this.storeDir, storeFilename)).toString();

        try {
            json = JSON.stringify(JSON.parse(json));
        }
        catch(error) {
            throw new Error(`Error at packStoredData(${storeFilename}): ${error.message}`);
        }

        const input = Buffer.from(json).toJSON().data;
        
        let fakeBytes = [];

        const step = 31;
        let i = 1;
        for (const byte of input) {
            i++;
            if (i > step)
                i=1;
            fakeBytes.push(byte + step + i);
        }

        return JSON.stringify(fakeBytes);
    }

    /*
    **
    **
    */
    transformObjectKeysToLitterals(js) {

        const ast = recast.parse(js);

        const isObject = function(test) { return typeof test === 'object' && test !== null }
        
        recast.visit(ast, {

            visitProperty(path) {
            
                if (isObject(path) 
                    && isObject(path.parentPath)
                    && isObject(path.parentPath.parentPath)
                    && isObject(path.parentPath.parentPath.value)
                    && path.parentPath.parentPath.value.type === 'ObjectExpression'
                    && isObject(path.value)
                    && isObject(path.value.key)
                    && path.value.key.type === 'Identifier')
                {
                    const keyName = path.value.key.name;
                    
                    path.value.key.type = 'Literal'
                    path.value.key.value = `["${keyName}"]`;
        	        path.value.key.raw = `["${keyName}"]`;
                } 

                return false;
            }
        });

        return recast.print(ast).code;
    }

    /*
    **
    **
    */
    runJSObfuscator(js) {

        const options = {
            compact: true,
            controlFlowFlattening: true,
            controlFlowFlatteningThreshold: 0.2,
            deadCodeInjection: false,
            deadCodeInjectionThreshold: 0.2,
            debugProtection: false,
            debugProtectionInterval: false,
            disableConsoleOutput: true,
            domainLock: [],
            identifierNamesGenerator: 'mangled',
            identifiersDictionary: [],
            identifiersPrefix: '',
            inputFileName: '',
            log: false,
            renameGlobals: true,
            renameProperties: false,
            reservedNames: [],
            reservedStrings: [],
            rotateStringArray: true,
            seed: 0,
            selfDefending: false,
            shuffleStringArray: true,
            sourceMap: false,
            sourceMapBaseUrl: '',
            sourceMapFileName: '',
            sourceMapMode: 'separate',
            splitStrings: false,
            splitStringsChunkLength: 4,
            stringArray: true,
            stringArrayEncoding: [
                'rc4',
                'base64'
            ],
            stringArrayThreshold: 1,
            stringArrayIndexShift: true,
            stringArrayRotate: true,
            stringArrayShuffle: true,
            stringArrayWrappersCount: 5,
            stringArrayWrappersChainedCalls: true,    
            stringArrayWrappersParametersMaxCount: 5,
            stringArrayWrappersType: 'function',
            target: 'browser',
            transformObjectKeys: true,
            unicodeEscapeSequence: false,
            simplify: true
        };

        const obf = JavaScriptObfuscator.obfuscate(js, options);
        
        return obf.getObfuscatedCode();
    }

    /*
    **
    **
    */
    async runTerser(js) {

        const options = {
            compress: true,
            mangle: {
                toplevel: true,
                properties: false
            }
        };

        const output = await Terser.minify(js, options);

        return output.code;
    }

    /*
    **
    **
    */
    runESTransform(js) {

        return esbuild.transformSync(js, {
            target: 'es6',
            minify: true,
            sourcemap: false,
            legalComments: 'none'
        }).code;
    }

    /*
    **
    **
    */
    async buildCSS() {

        fs.writeFileSync(this.cssPath, this.toCSS(this.mainScss, this.mode === 'dev'));
    }

    /*
    **
    **
    */
    toCSS(file, sourceMap) {

        const rendered = sass.renderSync({
            file: file,
            outputStyle: 'compressed',
            sourceMap: sourceMap ? '' : false,
            sourceMapEmbed: sourceMap,
            sourceMapContents: sourceMap,
            functions: {
                'svg($arg1)': function(arg1) {
                    try {
                        let content = fs.readFileSync(`${this.srcDir}${arg1.getValue()}`);
                        return new sass.types.String(`url("data:image/svg+xml;base64,${content.toString('base64')}")`);
                    }
                    catch(error) {
                        Log.red(`svg image not found: ${arg1.getValue()}`);
                        return new sass.types.String(`none`);
                    }
                }.bind(this)
            },
            includePaths: [
                nodePath.resolve(process.cwd(), 'node_modules')
            ]
        });

        return rendered.css;
    }

    /*
    **
    **
    */
    async buildHTML() {

        //========
        //LAUNCHER
        //========

        let launchScreenHTML;
        let launchScreenJS;
        let launchScreenCSS;
        
        try {

            launchScreenHTML = fs.readFileSync(nodePath.resolve(this.srcDir, 'launcher/launcher.html')).toString();
            launchScreenJS = fs.readFileSync(nodePath.resolve(this.srcDir, 'launcher/launcher.js')).toString();
            launchScreenCSS = this.toCSS(nodePath.resolve(this.srcDir, 'launcher/launcher.scss'), false);
        }
        catch(error) {
            
            throw new Error(`Launcher files not found`);
        }

        if (this.mode === 'dev') {
            launchScreenJS += ';' + fs.readFileSync(nodePath.resolve(process.cwd(), 'runtime/browser/dev-tool.js')).toString();
            launchScreenCSS += this.toCSS(nodePath.resolve(process.cwd(), 'runtime/browser/dev-tool.scss'), false);
        }

        launchScreenJS = this.runESTransform(launchScreenJS);

        if (this.mode === 'build')
            launchScreenJS = await this.runTerser(launchScreenJS);

        launchScreenJS = `<script>${launchScreenJS}</script>`;
        launchScreenCSS = `<style>${launchScreenCSS}</style>`;

        //=========
        //HTML HEAD
        //=========

        let deferredResources = JSON.stringify({
            css: this.cssLinks,
            js: this.jsLinks
        });

        deferredResources = `<script id="deferred-resources" type="application/json">${deferredResources}</script>`;

        const options = RuntimeData.getOptions();

        const replacements = {
            '{appName}': options.appName,
            '{themeColor}': options.themeColor,
            '{robots}': this.mode === 'build' && options.robotsOnProd === true ? 'index, follow' : 'noindex, nofollow',
            '{domain}': options.domain,
            '{og:type}': options['og:type'],
            '{twitterSite}': options['twitter:site'],
            '{extraHead}': options.extraHead,
            '{launchScreenCSS}': launchScreenCSS,
            '{launchScreenJS}': launchScreenJS,
            '{deferredResources}': deferredResources
        };
        
        for (let key in replacements)
            launchScreenHTML = launchScreenHTML.replace(new RegExp(key, 'g'), replacements[key] ? replacements[key] : '');
        
        //====
        //HTML
        //====

        launchScreenHTML = await this.minifyHTML(launchScreenHTML);

        fs.writeFileSync(nodePath.resolve(this.distModeDir, 'index.html'), launchScreenHTML);
    }

    /*
    **
    **
    */
    minifyHTML(html) {

        return new Promise(function(resolve, reject) {

            const parser = new minimizeHTML({ 
                quotes: true 
            });

            parser.parse(html, function(error, data) {

                if (error)
                    resolve(error.toString());
                else
                    resolve(data.replace(/^\s+|\s+$/g, ''));

            }.bind(this));

        }.bind(this));
    }
}
'use strict';

export class Log {

    static Reset = "\x1b[0m";
    static Bright = "\x1b[1m";
    static Dim = "\x1b[2m";
    static Underline = "\x1b[4m";
    static Blink = "\x1b[5m";
    static Reverse = "\x1b[7m";
    static Hidden = "\x1b[8m";

    static FgBlack = "\x1b[30m";
    static FgRed = "\x1b[31m";
    static FgGreen = "\x1b[32m";
    static FgYellow = "\x1b[33m";
    static FgBlue = "\x1b[34m";
    static FgMagenta = "\x1b[35m";
    static FgCyan = "\x1b[36m";
    static FgWhite = "\x1b[37m";

    static BgBlack = "\x1b[40m";
    static BgRed = "\x1b[41m";
    static BgGreen = "\x1b[42m";
    static BgYellow = "\x1b[43m";
    static BgBlue = "\x1b[44m";
    static BgMagenta = "\x1b[45m";
    static BgCyan = "\x1b[46m";
    static BgWhite = "\x1b[47m";

    /*
    **
    **
    */
    static log(message, currentProcess = true) {

        if (currentProcess)
            message = `${Log.Dim}[${new Date().toLocaleString('fr', { timeZone: process.env.TIME_ZONE })}] @${process.pid}${Log.Reset} [${process.env.LOG_ID}] ${message}`;
        
        const lines = message.split('\n');

        for (const line of lines) {

            if (line.length === 0)
                continue;
            
            process.stdout.write(`${line}\n`);
        }
    }

    /*
    **
    **
    */
    static blank() {

        Log.log('');
    }

    /*
    **
    **
    */
    static green(message) {

        Log.log(`${Log.FgGreen}${Log.Bright}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    static red(message) {

        Log.log(`${Log.FgRed}${Log.Bright}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    static yellow(message) {

        Log.log(`${Log.FgYellow}${Log.Bright}${message}${Log.Reset}`);
    }

    /*
    **
    **
    */
    static printError(error) {

        if (typeof error !== 'object' || error === null || !error.stack)
            return;

        error.stack.split('\n').forEach((line) => {
            Log.log(`${Log.FgRed}${line}${Log.Reset}`);
        });
    }
}
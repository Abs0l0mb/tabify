import {
    Block,
    Div,
    EditTaskPopup,
    DeleteTaskPopup,
    Tools,
    AltBox,
    TasksColumn,
    ClientLocation
} from '@src/classes';

export interface TaskData {
    id: number,
    team_id: number,
    title: string,
    description: string,
    category_id: number,
    category_title: string,
    estimated_hours: number,
    executor_account_id: number;
    executor: string,
    creator_account_id: number;
    creator: string
    status: string,
    time: string
}

export class Task extends Div {

    private dataContainer: Div;

    private title: Div;
    private description: AltBox;
    private category: Div;
    private estimatedHours: Div;
    private since: Div | null = null;

    private lastSinceValue: string | null = null;

    constructor(private data: TaskData, private column: TasksColumn, parent?: Block) {

        super('task task-kanban', parent);

        this.setData('status', column.getStatus());

        //=========
        //GRAB ZONE
        //=========

        if (['PREDEFINED', 'TODO'].includes(column.getStatus())
        || (column.getStatus() === 'IN_PROGRESS' && data.executor_account_id === ClientLocation.get().api.accountData.id)) {

            this.setData('grabbable', 1);

            new Div('grab', this).onNative('mousedown', this.onGrabZoneMouseDown.bind(this));
        }

        //==============
        //DATA CONTAINER
        //==============

        this.dataContainer = new Div('data-container', this);

        //==============================
        //TITLE (AND ALTBOX DESCRIPTION)
        //==============================

        this.title = new Div('title', this.dataContainer).write(data.title);

        this.description = new AltBox(this.title, data.description);

        //=================
        //SIMPLE KEY VALUES
        //=================

        this.category = this.addKeyValue('Category', data.category_title, 'category');

        if (data.creator)
            this.addKeyValue('Creator', data.creator, 'actor');

        if (data.executor)
            this.addKeyValue('Executor', data.executor, 'actor');
        
        this.estimatedHours = this.addKeyValue('Estimated hours', data.estimated_hours.toString(), 'estimated-hours');

        if (column.getStatus() !== 'PREDEFINED') {
            this.since = this.addKeyValue(`${column.getBeautifiedStatus()} since`, '', 'since');
            this.updateSinceValue();
        }

        //=========================
        //UPDATE AND DELETE BUTTONS
        //=========================

        if (['PREDEFINED', 'TODO'].includes(column.getStatus())) {

            new Div('update', this.dataContainer).onNative('click', this.onUpdate.bind(this));
            new Div('delete', this.dataContainer).onNative('click', this.onDelete.bind(this));
        }
        
        //=============
        //ASYNC DISPLAY
        //=============

        this.display();
    }

    /*
    **
    **
    */
    private addKeyValue(key: string, value: string, className?: string) : Div {

        const containerDiv = new Div('key-value', this.dataContainer);
        const keyDiv = new Div('key', containerDiv).write(`${key}:`);
        const valueDiv = new Div('value', containerDiv).write(value);

        if (className)
            containerDiv.addClass(className);

        return valueDiv;
    }

    /*
    **
    **
    */
    public async display() : Promise<void> {

        await Tools.sleep(10);
        this.setData('displayed', 1);
    }
    
    /*
    **
    **
    */
    public syncDisplay() : void {

        this.setData('displayed', 1);
    }

    /*
    **
    **
    */
    public syncHide() : void {

        this.setData('displayed', -1);
    }

    /*
    **
    **
    */
    public async setTranslucent() : Promise<void> {

        await Tools.sleep(10);
        this.setData('displayed', 2);
    }

    /*
    **
    **
    */
    private onGrabZoneMouseDown() : void {

        this.emit('grabbed-task', this);
    }

    /*
    **
    **
    */
    private onUpdate() : void {

        new EditTaskPopup(this.data.id, this.data.team_id);
    }

    /*
    **
    **
    */
    private onDelete() : void {

        new DeleteTaskPopup(this.data.id);
    }

    /*
    **
    **
    */
    public getData() : TaskData {

        return this.data;
    }

    /*
    **
    **
    */
    public getColumn() : TasksColumn {

        return this.column;
    }

    /*
    **
    **
    */
    public updateSinceValue() : void {

        if (!this.since)
            return;
        
        let sinceValue = Tools.humanPeriod(Date.now() - new Date(this.data.time).getTime());

        if (this.lastSinceValue !== sinceValue)
            this.since.write(sinceValue);

        this.lastSinceValue = sinceValue;
    }
}